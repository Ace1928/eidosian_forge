import sqlite3
from typing import List, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class DataAnalysis:
    """
    Provides functionality for performing advanced data analytics on stored embeddings.
    This class uses SQLite for data storage, sklearn for data analysis, and numpy for data manipulation.
    """

    def __init__(self, analysis_db_path: str):
        """
        Initializes connections to the analysis database.
        :param analysis_db_path: str - Path to the SQLite database file where analysis results are stored.
        """
        self.db_path = analysis_db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Creates the necessary tables in the database if they do not already exist, structured to store various types of analysis results.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pca_results (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                component_1 FLOAT,
                component_2 FLOAT
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tsne_results (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                component_1 FLOAT,
                component_2 FLOAT
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clustering_results (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                cluster_id INTEGER
            )
        """
        )
        self.connection.commit()

    def _perform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Performs PCA on the given embeddings to reduce dimensionality to 2 components for visualization.
        :param embeddings: np.ndarray - A numpy array of embeddings.
        :return: np.ndarray - The result of PCA dimensionality reduction.
        """
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings)

    def _perform_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Performs t-SNE on the given embeddings to reduce dimensionality to 2 components for detailed visualization.
        :param embeddings: np.ndarray - A numpy array of embeddings.
        :return: np.ndarray - The result of t-SNE dimensionality reduction.
        """
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(embeddings)

    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Applies K-means clustering to the embeddings to identify clusters within the data.
        :param embeddings: np.ndarray - A numpy array of embeddings.
        :return: np.ndarray - The cluster labels for each embedding.
        """
        kmeans = KMeans(n_clusters=5, random_state=42)
        return kmeans.fit_predict(embeddings)

    def perform_analysis(self, embeddings: List[Tuple[str, np.ndarray]]) -> None:
        """
        Performs various types of data analysis on the embeddings and stores results in separate database tables.
        :param embeddings: List[Tuple[str, np.ndarray]] - A list of tuples containing file paths and their numpy array embeddings.
        """
        embeddings_array = np.array([emb[1] for emb in embeddings])
        pca_results = self._perform_pca(embeddings_array)
        tsne_results = self._perform_tsne(embeddings_array)
        cluster_labels = self._perform_clustering(embeddings_array)

        for idx, (file_path, _) in enumerate(embeddings):
            self.cursor.execute(
                "INSERT INTO pca_results (file_path, component_1, component_2) VALUES (?, ?, ?)",
                (file_path, pca_results[idx][0], pca_results[idx][1]),
            )
            self.cursor.execute(
                "INSERT INTO tsne_results (file_path, component_1, component_2) VALUES (?, ?, ?)",
                (file_path, tsne_results[idx][0], tsne_results[idx][1]),
            )
            self.cursor.execute(
                "INSERT INTO clustering_results (file_path, cluster_id) VALUES (?, ?)",
                (file_path, cluster_labels[idx]),
            )

        self.connection.commit()

    def __del__(self):
        """
        Ensures the database connection is closed when the object is deleted.
        """
        self.connection.close()
