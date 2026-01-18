import sqlite3
from typing import List, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import logging
from embedding_storage import (
    EmbeddingStorage,
)  # Importing the EmbeddingStorage class


class DataAnalysis:
    """
    Handles advanced data analytics on embeddings stored in a SQLite database. This includes PCA, t-SNE, and K-means clustering.
    Utilizes the EmbeddingStorage class to retrieve embeddings for analysis.
    """

    def __init__(self, analysis_db_path: str, embedding_db_path: str):
        """
        Connects to the SQLite database at the specified path for analysis and utilizes the EmbeddingStorage class to interact with the embeddings database.
        :param analysis_db_path: Path to the SQLite database file for storing analysis results.
        :param embedding_db_path: Path to the SQLite database file where embeddings are stored.
        """
        self.analysis_db_path = analysis_db_path
        self.embedding_storage = EmbeddingStorage(
            embedding_db_path, None
        )  # Initialize EmbeddingStorage without a folder path
        self.connection = sqlite3.connect(self.analysis_db_path)
        self.cursor = self.connection.cursor()
        self._create_tables()
        logging.basicConfig(level=logging.INFO)

    def _create_tables(self):
        """Creates tables in the database for storing analysis results."""
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS pca_results (id INTEGER PRIMARY KEY, file_path TEXT, component_1 FLOAT, component_2 FLOAT)"""
        )
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS tsne_results (id INTEGER PRIMARY KEY, file_path TEXT, component_1 FLOAT, component_2 FLOAT)"""
        )
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS clustering_results (id INTEGER PRIMARY KEY, file_path TEXT, cluster_id INTEGER)"""
        )
        self.connection.commit()

    def _perform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies PCA to reduce dimensionality of embeddings to two components."""
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings)

    def _perform_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies t-SNE for detailed visualization by reducing dimensionality to two components."""
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(embeddings)

    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Identifies clusters within the embeddings using K-means clustering."""
        kmeans = KMeans(n_clusters=5, random_state=42)
        return kmeans.fit_predict(embeddings)

    def perform_analysis(self):
        """Performs PCA, t-SNE, and clustering on the embeddings retrieved from the EmbeddingStorage and stores the results in the database."""
        embeddings_data = self.embedding_storage.load_embeddings()
        embeddings_array = np.array([emb[1] for emb in embeddings_data])
        pca_results = self._perform_pca(embeddings_array)
        tsne_results = self._perform_tsne(embeddings_array)
        cluster_labels = self._perform_clustering(embeddings_array)

        for idx, (file_path, _) in enumerate(embeddings_data):
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
        """Closes the database connection when the object is deleted."""
        self.connection.close()
        logging.info("Database connection closed.")
