import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Função para carregar e preparar dados
def carregar_dados(filepath):
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y, %H:%M:%S')
    df['deathDate'] = pd.to_datetime(df['deathDate'], format='%d/%m/%Y, %H:%M:%S', errors='coerce')
    df.rename(columns={'main_reason': 'motivos'}, inplace=True)
    return df

# Carregar dados e selecionar os 35 bairros principais
df = carregar_dados('fogo_cruzado_2022_2024.csv')
top_bairros = df['neighborhood'].value_counts().nlargest(35)
df = df[df['neighborhood'].isin(top_bairros.index)]

# Gráfico 1: Número de Incidentes por Bairro (Ordem Decrescente e com 35 Bairros)
top_bairros = df['neighborhood'].value_counts().nlargest(35)  # Seleciona os 35 bairros com mais incidentes
df_top_bairros = pd.DataFrame({'Bairro': top_bairros.index, 'Número de Incidentes': top_bairros.values})

fig1 = px.bar(
    df_top_bairros,
    x='Número de Incidentes', 
    y='Bairro',
    orientation='h',
    title="Número de Incidentes por Bairro",
    labels={'Número de Incidentes': 'Número de Incidentes', 'Bairro': 'Bairro'},
    hover_data={'Número de Incidentes': True}  # Remove o campo 'color'
)
fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=1000)  # Ajustar a altura do gráfico
fig1.show()

# Gráfico 2: Tipos de Ocorrências por Bairro (Ordem Decrescente e com 35 Bairros)
df_grouped = df.groupby(['neighborhood', 'motivos']).size().unstack(fill_value=0)
df_grouped = df_grouped.loc[top_bairros.index]  # Garante a ordem decrescente dos bairros mais incidentes

fig2 = go.Figure()
for motivo in df_grouped.columns:
    fig2.add_trace(go.Bar(
        y=df_grouped.index,
        x=df_grouped[motivo],
        name=motivo,
        orientation='h'
    ))

fig2.update_layout(
    barmode='stack', 
    title='Tipos de Ocorrências por Bairro',
    yaxis={'categoryorder': 'total ascending'},
    xaxis_title='Número de Ocorrências',
    yaxis_title='Bairro',
    height=1000  # Ajustar a altura do gráfico
)
fig2.show()

# Engenharia de recursos para modelagem
def engenharia_de_recursos(df):
    df['hour'] = df['data'].dt.hour
    df['day_of_week'] = df['data'].dt.dayofweek
    df['month'] = df['data'].dt.month
    return df

df = engenharia_de_recursos(df)

# Preparar dados para classificação
def preparar_dados(df):
    features = ['latitude', 'longitude', 'hour', 'day_of_week', 'month', 'age']
    X = df[features]
    y = df['motivos']

    le = LabelEncoder()
    y = le.fit_transform(y)

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le

X, y, le = preparar_dados(df)

# Tratamento de desequilíbrio de classes
def tratar_desequilibrio(X, y):
    # Remover classes com menos de 2 amostras
    class_counts = np.bincount(y)
    valid_classes = np.where(class_counts >= 2)[0]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    smote = SMOTE(random_state=42, k_neighbors=1)
    return smote.fit_resample(X, y)

X_res, y_res = tratar_desequilibrio(X, y)

# Classificação das Motivações dos Incidentes
def treinar_modelo(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
model = treinar_modelo(X_train, y_train)

# Avaliação do modelo de classificação
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Acurácia:", accuracy_score(y_test, y_pred))

# Agrupamento dos Incidentes por Similaridade (K-Means)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

df['Cluster'] = clusters  # Atribuir clusters ao DataFrame

# Visualização dos clusters com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig3 = px.scatter(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    color=df['Cluster'].astype(str),
    title='Visualização dos Clusters com PCA',
    labels={'x': 'Componente Principal 1', 'y': 'Componente Principal 2', 'color': 'Cluster'}
)
fig3.show()

# Novo Gráfico: Ocorrências por Dia da Semana e Horário
def grafico_ocorrencias_dia_hora(df):
    # Traduzir os dias da semana para português
    df['dia_semana'] = df['data'].dt.day_name().map({
        'Monday': 'Segunda-feira',
        'Tuesday': 'Terça-feira',
        'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira',
        'Friday': 'Sexta-feira',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    })
    df['hora'] = df['data'].dt.hour
    ocorrencias_dia_hora = df.groupby(['dia_semana', 'hora']).size().reset_index(name='ocorrencias')
    fig = px.density_heatmap(ocorrencias_dia_hora, x='hora', y='dia_semana', z='ocorrencias', 
                             title='Ocorrências por Dia da Semana e Horário', 
                             labels={'hora': 'Hora do Dia', 'dia_semana': 'Dia da Semana', 'ocorrencias': 'Número de Ocorrências'},
                             color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1  # Detalhar mais os horários no eixo x
        ),
        yaxis=dict(
            categoryorder='array',
            categoryarray=['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        ),
        coloraxis_colorbar=dict(
            title='Número de Ocorrências'  # Remover "sum of"
        )
    )
    fig.update_traces(
        hovertemplate='Hora: %{x}:00-%{x}:59<br>Dia: %{y}<br>Número de Ocorrências: %{z}<extra></extra>'
    )
    return fig

fig4 = grafico_ocorrencias_dia_hora(df)
fig4.show()

# Gráfico 5: Ocorrências ao Longo do Tempo por Bairro
def grafico_ocorrencias_tempo_bairro(df):
    df['ano_mes'] = df['data'].dt.to_period('M').astype(str)  # Converter Period para string
    ocorrencias_tempo_bairro = df.groupby(['ano_mes', 'neighborhood']).size().reset_index(name='ocorrencias')
    fig = px.line(ocorrencias_tempo_bairro, x='ano_mes', y='ocorrencias', color='neighborhood',
                  title='Ocorrências ao Longo do Tempo por Bairro',
                  labels={'ano_mes': 'Ano e Mês', 'ocorrencias': 'Número de Ocorrências', 'neighborhood': 'Bairro'})
    return fig

fig5 = grafico_ocorrencias_tempo_bairro(df)
fig5.show()

# Gráfico 6: Ocorrências por Tipo de Crime e Bairro
def grafico_ocorrencias_tipo_bairro(df):
    ocorrencias_tipo_bairro = df.groupby(['motivos', 'neighborhood']).size().reset_index(name='ocorrencias')
    fig = px.bar(ocorrencias_tipo_bairro, x='neighborhood', y='ocorrencias', color='motivos',
                 title='Ocorrências por Tipo de Crime e Bairro',
                 labels={'neighborhood': 'Bairro', 'ocorrencias': 'Número de Ocorrências', 'motivos': 'Tipo de Crime'})
    return fig

fig6 = grafico_ocorrencias_tipo_bairro(df)
fig6.show()

# Gráfico 7: Mapa de Calor de Ocorrências por Latitude e Longitude
def grafico_mapa_calor(df):
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='motivos', radius=10,
                            center=dict(lat=-12.9714, lon=-38.5014), zoom=10,
                            mapbox_style="stamen-terrain",
                            title='Mapa de Calor de Ocorrências por Latitude e Longitude')
    return fig

fig7 = grafico_mapa_calor(df)
fig7.show()

# Gráfico 8: Idade das Vítimas por Tipo de Crime
def grafico_idade_tipo_crime(df):
    fig = px.scatter(df, x='age', y='motivos', color='motivos',
                     title='Idade das Vítimas por Tipo de Crime',
                     labels={'age': 'Idade', 'motivos': 'Tipo de Crime'})
    return fig

fig8 = grafico_idade_tipo_crime(df)
fig8.show()

# Gráfico 9: Ocorrências por Mês e Tipo de Crime
def grafico_ocorrencias_mes_tipo(df):
    ocorrencias_mes_tipo = df.groupby(['month', 'motivos']).size().reset_index(name='ocorrencias')
    fig = px.bar(ocorrencias_mes_tipo, x='month', y='ocorrencias', color='motivos',
                 title='Ocorrências por Mês e Tipo de Crime',
                 labels={'month': 'Mês', 'ocorrencias': 'Número de Ocorrências', 'motivos': 'Tipo de Crime'})
    return fig

fig9 = grafico_ocorrencias_mes_tipo(df)
fig9.show()