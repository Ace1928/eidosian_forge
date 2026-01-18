import os
import hashlib
import ast
import json
import logging
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import entropy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Canvas
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configure logging with a detailed format specification
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DatabaseConnectionManager:
    """
    Manages the lifecycle of database connections, ensuring robust and reusable database connectivity.
    This class encapsulates connection handling and transaction management, serving as a foundational component for database interactions.
    """

    def __init__(self, database_path: str):
        """
        Initializes a DatabaseConnectionManager instance with a specified path to the SQLite database.

        Parameters:
        - database_path (str): The filesystem path to the SQLite database file.
        """
        self._database_path: str = database_path
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

    def __enter__(self) -> "DatabaseConnectionManager":
        """
        Establishes a database connection upon entering the runtime context, preparing the manager for database operations.

        Returns:
        - DatabaseConnectionManager: The instance with an active database connection.
        """
        self._connection = sqlite3.connect(self._database_path)
        self._cursor = self._connection.cursor()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[type],
    ) -> None:
        """
        Terminates the database connection upon exiting the runtime context, ensuring clean disconnection.

        Parameters:
        - exc_type (Optional[type]): The type of the exception, if any.
        - exc_val (Optional[Exception]): The exception instance, if any.
        - exc_tb (Optional[type]): The traceback object, if any.
        """
        if self._connection:
            self._connection.close()

    def commit_transactions(self) -> None:
        """
        Commits the current transaction to the database, ensuring that all changes made during the transaction are persisted.

        This method is critical for maintaining data integrity and consistency within the database.
        """
        if self._connection:
            self._connection.commit()


class EmbeddingDatabaseManager(DatabaseManager):
    """
    Manages the database operations specifically for embeddings, including initialization, insertion, retrieval, and closure.
    """

    def initialize_database(self) -> None:
        """
        Initialize the database by creating the necessary table if it does not exist.
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                file_hash TEXT UNIQUE,
                imports TEXT,
                function TEXT,
                docstring TEXT,
                embedding BLOB
            )
        """
        self.cursor.execute(create_table_query)
        self.commit_changes()

    def insert_embedding(
        self,
        file_hash: str,
        imports: str,
        function: str,
        docstring: str,
        embedding: bytes,
    ) -> None:
        """
        Insert an embedding into the database.

        :param file_hash: str - The hash of the file.
        :param imports: str - The imports used in the file.
        :param function: str - The function code.
        :param docstring: str - The docstring of the function.
        :param embedding: bytes - The serialized embedding.
        """
        insert_query = """
            INSERT INTO embeddings (file_hash, imports, function, docstring, embedding) 
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            self.cursor.execute(
                insert_query, (file_hash, imports, function, docstring, embedding)
            )
        except sqlite3.DatabaseError as e:
            logging.error(f"Error inserting embedding: {e}")
            raise

    def is_file_processed(self, file_hash: str) -> bool:
        """
        Check if a file has already been processed by querying its hash.

        :param file_hash: str - The hash of the file to check.
        :return: bool - True if the file has been processed, False otherwise.
        """
        check_query = "SELECT id FROM embeddings WHERE file_hash = ?"
        try:
            self.cursor.execute(check_query, (file_hash,))
            return self.cursor.fetchone() is not None
        except sqlite3.DatabaseError as e:
            logging.error(f"Error checking processed file: {e}")
            raise

    def retrieve_embeddings(
        self, search_term: str
    ) -> list[tuple[str, str, str, bytes]]:
        """
        Retrieve embeddings that match the search term either in function or docstring.

        :param search_term: str - The term to search for in function names or docstrings.
        :return: list[tuple[str, str, str, bytes]] - A list of tuples containing imports, function, docstring, and embedding.
        """
        retrieve_query = """
            SELECT imports, function, docstring, embedding 
            FROM embeddings 
            WHERE function LIKE ? OR docstring LIKE ? COLLATE NOCASE
        """
        try:
            self.cursor.execute(
                retrieve_query, ("%" + search_term + "%", "%" + search_term + "%")
            )
            return self.cursor.fetchall()
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving embeddings: {e}")
            raise

    def retrieve_all_embeddings(self) -> list[tuple[str, str, str, bytes]]:
        """
        Retrieve all embeddings from the database using pagination to handle large datasets efficiently.

        :return: list[tuple[str, str, str, bytes]] - A list of tuples containing imports, function, docstring, and embedding.
        """
        retrieve_all_query = (
            "SELECT imports, function, docstring, embedding FROM embeddings"
        )
        try:
            self.cursor.execute(retrieve_all_query)
            results = []
            while True:
                page = self.cursor.fetchmany(1000)  # Fetch results in pages of 1000
                if not page:
                    break
                results.extend(page)
            return results
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving all embeddings: {e}")
            raise


class DataAnalysisDatabaseManager(DatabaseManager):
    """
    Manages the database operations specifically for data analysis results, including the initialization of tables,
    storage of analysis results, and retrieval of these results.
    """

    def initialize_database(self) -> None:
        """
        Initialize the database by creating the 'analysis' table if it does not already exist.
        This table is structured to store analysis type, parameters used, and the results in binary format.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY,
            analysis_type TEXT,
            parameters TEXT,
            results BLOB
        )
        """
        self.cursor.execute(create_table_query)
        self.commit_changes()

    def store_analysis_results(
        self, analysis_type: str, parameters: str, results: bytes
    ) -> None:
        """
        Store the results of a specific analysis in the database.

        :param analysis_type: str - The type of analysis performed.
        :param parameters: str - The parameters used for the analysis.
        :param results: bytes - The binary results of the analysis.
        """
        insert_query = (
            "INSERT INTO analysis (analysis_type, parameters, results) VALUES (?, ?, ?)"
        )
        try:
            self.cursor.execute(insert_query, (analysis_type, parameters, results))
        except sqlite3.DatabaseError as e:
            logging.error(f"Error storing analysis results: {e}")
            raise

    def retrieve_analysis_results(
        self, analysis_type: str, parameters: str
    ) -> bytes | None:
        """
        Retrieve the results of a specific analysis from the database based on the analysis type and parameters.

        :param analysis_type: str - The type of analysis to retrieve.
        :param parameters: str - The parameters used for the analysis.
        :return: bytes | None - The results of the analysis or None if not found.
        """
        select_query = (
            "SELECT results FROM analysis WHERE analysis_type = ? AND parameters = ?"
        )
        try:
            self.cursor.execute(select_query, (analysis_type, parameters))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving analysis results: {e}")
            raise


class DataAnalysisManager:
    """
    Manages data analysis operations on embeddings, providing methods for clustering,
    dimensionality reduction, and other analytical techniques.
    """

    def __init__(self, database_path: str):
        """
        Initialize the DataAnalysisManager with a connection to the specified database.

        :param database_path: str - The path to the database file.
        """
        self.database_path = database_path
        self.database_manager = DataAnalysisDatabaseManager(self.database_path)
        self.database_manager.initialize_database()

    def perform_kmeans_clustering(
        self, embeddings: np.ndarray, num_clusters: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform K-means clustering on the given embeddings.

        :param embeddings: np.ndarray - The embeddings to cluster.
        :param num_clusters: int - The number of clusters to form.
        :return: tuple[np.ndarray, np.ndarray] - The labels and centroids of the clusters.
        """
        try:
            kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
            return kmeans.labels_, kmeans.cluster_centers_
        except ValueError as error:
            logging.error(f"K-means clustering failed: {error}")
            raise

    def perform_agglomerative_clustering(
        self, embeddings: np.ndarray, num_clusters: int
    ) -> np.ndarray:
        """
        Perform Agglomerative clustering on the given embeddings.

        :param embeddings: np.ndarray - The embeddings to cluster.
        :param num_clusters: int - The number of clusters to form.
        :return: np.ndarray - The labels of the clusters.
        """
        try:
            agglomerative = AgglomerativeClustering(n_clusters=num_clusters).fit(
                embeddings
            )
            return agglomerative.labels_
        except ValueError as error:
            logging.error(f"Agglomerative clustering failed: {error}")
            raise

    def perform_dbscan_clustering(
        self, embeddings: np.ndarray, epsilon: float, min_samples: int
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering on the given embeddings.

        :param embeddings: np.ndarray - The embeddings to cluster.
        :param epsilon: float - The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param min_samples: int - The number of samples in a neighborhood for a point to be considered as a core point.
        :return: np.ndarray - The labels of the clusters.
        """
        try:
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(embeddings)
            return dbscan.labels_
        except ValueError as error:
            logging.error(f"DBSCAN clustering failed: {error}")
            raise

    def perform_pca(self, embeddings: np.ndarray, components: int) -> np.ndarray:
        """
        Perform Principal Component Analysis (PCA) on the given embeddings to reduce their dimensionality.

        :param embeddings: np.ndarray - The embeddings to transform.
        :param components: int - The number of components to keep.
        :return: np.ndarray - The transformed embeddings.
        """
        try:
            pca = PCA(n_components=components)
            return pca.fit_transform(embeddings)
        except ValueError as error:
            logging.error(f"PCA failed: {error}")
            raise

    def perform_tsne(self, embeddings: np.ndarray, components: int) -> np.ndarray:
        """
        Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) on the given embeddings to reduce their dimensionality.

        :param embeddings: np.ndarray - The embeddings to transform.
        :param components: int - The number of components to keep.
        :return: np.ndarray - The transformed embeddings.
        """
        try:
            if components >= len(embeddings):
                raise ValueError(
                    "t-SNE perplexity must be less than the number of samples"
                )
            tsne = TSNE(n_components=components)
            return tsne.fit_transform(embeddings)
        except ValueError as error:
            logging.error(f"t-SNE failed: {error}")
            raise

    def calculate_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate the pairwise distances between all pairs of embeddings.

        :param embeddings: np.ndarray - The embeddings to calculate distances between.
        :return: np.ndarray - The matrix of distances.
        """
        try:
            return cdist(embeddings, embeddings)
        except ValueError as error:
            logging.error(f"Calculating pairwise distances failed: {error}")
            raise

    def build_kdtree(self, embeddings: np.ndarray) -> KDTree:
        """
        Build a KDTree from the given embeddings for efficient spatial queries.

        :param embeddings: np.ndarray - The embeddings to build the KDTree with.
        :return: KDTree - The constructed KDTree.
        """
        try:
            return KDTree(embeddings)
        except ValueError as error:
            logging.error(f"Building KDTree failed: {error}")
            raise

    def calculate_entropy(self, labels: np.ndarray) -> float:
        """
        Calculate the entropy of the given labels, providing a measure of the randomness in the data.

        :param labels: np.ndarray - The labels for which to calculate entropy.
        :return: float - The calculated entropy value.
        """
        try:
            _, counts = np.unique(labels, return_counts=True)
            probabilities = counts / len(labels)
            return entropy(probabilities)
        except ValueError as error:
            logging.error(f"Calculating entropy failed: {error}")
            raise

    def store_analysis_results(self, **results) -> None:
        """
        Store the analysis results in the database.

        :param results: dict - A dictionary containing all analysis results.
        """
        try:
            with self.database_manager as db:
                for key, value in results.items():
                    db.store_analysis(key, "", pickle.dumps(value))
        except sqlite3.DatabaseError as error:
            logging.error(f"Storing analysis results failed: {error}")
            raise

    def fetch_analysis_results(self, analysis_type: str) -> Any:
        """
        Fetch the analysis results from the database.

        :param analysis_type: str - The type of analysis results to fetch.
        :return: Any - The fetched analysis results.
        """
        try:
            with self.database_manager as db:
                results = db.fetch_analysis(analysis_type, "")
                if results is None:
                    logging.warning(f"No analysis results found for {analysis_type}")
                    return None
                return pickle.loads(results)
        except sqlite3.DatabaseError as error:
            logging.error(f"Fetching analysis results failed: {error}")
            raise

    def fetch_all_analysis_results(self) -> dict[str, Any]:
        """
        Fetch all analysis results from the database.

        :return: dict[str, Any] - A dictionary containing all analysis results.
        """
        analysis_types = [
            "kmeans_labels",
            "centroids",
            "agglomerative_labels",
            "dbscan_labels",
            "similarity_matrix",
            "pca_embeddings",
            "tsne_embeddings",
            "pairwise_distances",
            "kdtree",
            "entropy_value",
        ]
        results = {}
        for analysis_type in analysis_types:
            result = self.fetch_analysis_results(analysis_type)
            if result is not None:
                results[analysis_type] = result
        return results


class FileManager:
    """
    Manages file operations related to processed and embedded files, including loading, saving, and processing files.
    """

    def __init__(self, processed_files_path: str, embedded_files_path: str):
        """
        Initializes the FileManager with paths to processed and embedded files.

        :param processed_files_path: str - Path to the JSON file containing processed files metadata.
        :param embedded_files_path: str - Path to the JSON file containing embedded files metadata.
        """
        self.processed_files_path = processed_files_path
        self.embedded_files_path = embedded_files_path
        self.processed_files = self._load_json(self.processed_files_path)
        self.embedded_files = self._load_json(self.embedded_files_path)

    def _load_json(self, path: str) -> dict:
        """
        Loads a JSON file from the specified path.

        :param path: str - Path to the JSON file.
        :return: dict - The loaded JSON object, or an empty dictionary if the file does not exist.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
            return {}

    def save_json(self, data: dict, path: str) -> None:
        """
        Saves a dictionary as a JSON file to the specified path.

        :param data: dict - The data to save.
        :param path: str - Path to save the JSON file.
        """
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            logging.error(f"Error saving JSON to {path}: {e}")
            raise

    def process_folder(self, folder_path: str, callback) -> Tuple[int, int]:
        """
        Processes all Python files in a specified folder using a callback function.

        :param folder_path: str - Path to the folder containing Python files.
        :param callback: Callable - A callback function to process each Python file.
        :return: Tuple[int, int] - Total number of Python files and number of processed files.
        """
        total_files = 0
        processed_files = 0
        for root, dirs, files in os.walk(folder_path):
            python_files = [file for file in files if file.endswith(".py")]
            total_files += len(python_files)
            for file in python_files:
                file_path = os.path.join(root, file)
                callback(file_path)
                processed_files += 1
        return total_files, processed_files

    def process_file(self, file_path: str, model, db_manager) -> None:
        """
        Processes a single Python file, extracting functions and imports, and storing embeddings in a database.

        :param file_path: str - Path to the Python file.
        :param model: SentenceTransformer - The model used to generate embeddings.
        :param db_manager: EmbeddingDatabaseManager - The database manager to store embeddings.
        """
        try:
            with open(file_path, "r") as file:
                content = file.read()
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            if not db_manager.is_file_processed(file_hash):
                functions, imports, classes = CodeParser.extract_functions_and_imports(
                    content
                )
                embeddings = [
                    model.encode(func["function"], convert_to_tensor=True).numpy()
                    for func in functions
                ]
                for func, embed in zip(functions, embeddings):
                    db_manager.insert_embedding(
                        file_hash,
                        imports,
                        func["function"],
                        func["docstring"],
                        embed.tobytes(),
                    )
        except IOError as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise


class CodeParser:
    """
    A class dedicated to parsing Python code to extract functions, imports, and classes.
    """

    @staticmethod
    def parse_python_content(content: str) -> ast.Module:
        """
        Parses the content of a Python file into an Abstract Syntax Tree (AST).

        :param content: str - The content of the Python file.
        :return: ast.Module - The AST representation of the content.
        """
        return ast.parse(content)

    @staticmethod
    def extract_functions_from_ast(tree: ast.Module) -> List[Dict[str, str]]:
        """
        Extracts functions from the AST of a Python file.

        :param tree: ast.Module - The AST of the Python file.
        :return: List[Dict[str, str]] - A list of dictionaries containing function code and docstrings.
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node) or ""
                function_code = ast.unparse(node)
                functions.append({"function": function_code, "docstring": docstring})
        return functions

    @staticmethod
    def extract_import_statements_from_ast(tree: ast.Module) -> str:
        """
        Extracts all import statements from the AST of a Python file.

        :param tree: ast.Module - The AST of the Python file.
        :return: str - A single string containing all import statements.
        """
        imports = " ".join(
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        return imports

    @staticmethod
    def extract_class_definitions_from_ast(tree: ast.Module) -> List[str]:
        """
        Extracts all class definitions from the AST of a Python file.

        :param tree: ast.Module - The AST of the Python file.
        :return: List[str] - A list of strings, each representing a class definition.
        """
        classes = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        return classes

    @staticmethod
    def extract_functions_imports_classes_from_content(
        content: str,
    ) -> Tuple[List[Dict[str, str]], str, List[str]]:
        """
        Extracts functions, imports, and classes from the content of a Python file.

        :param content: str - The content of the Python file.
        :return: Tuple[List[Dict[str, str]], str, List[str]] - A tuple containing a list of functions, a string of imports, and a list of classes.
        """
        tree = CodeParser.parse_python_content(content)
        functions = CodeParser.extract_functions_from_ast(tree)
        imports = CodeParser.extract_import_statements_from_ast(tree)
        classes = CodeParser.extract_class_definitions_from_ast(tree)
        return functions, imports, classes


class DataVisualizationManager:
    """
    Manages the visualization of data using PCA and advanced analysis results.
    This class encapsulates the functionality for both basic and advanced data visualization,
    ensuring modularity and high cohesion.
    """

    def __init__(self, data_analysis_manager: DataAnalysisManager):
        """
        Initializes the DataVisualizationManager with a reference to the DataAnalysisManager.

        :param data_analysis_manager: DataAnalysisManager - The manager responsible for data analysis operations.
        """
        self.data_analysis_manager = data_analysis_manager

    def visualize_basic_data(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        Visualizes data using PCA by plotting embeddings on a canvas with interactivity.

        :param embeddings: np.ndarray - The embeddings to visualize.
        :param labels: np.ndarray - The labels corresponding to each embedding.
        """
        try:
            pca_embeddings = self.data_analysis_manager.perform_pca(
                embeddings, n_components=2
            )
            visualization_window = tk.Toplevel()
            visualization_window.title("Embedding Visualization")
            canvas = tk.Canvas(visualization_window, width=800, height=800)
            canvas.pack()

            normalized_embeddings = self._normalize_embeddings(pca_embeddings)
            self._plot_embeddings(canvas, normalized_embeddings, labels)
            canvas.bind(
                "<Button-1>",
                lambda event: self._on_click(event, normalized_embeddings, labels),
            )

        except Exception as e:
            logging.error(f"Error visualizing basic data: {e}")
            tk.messagebox.showerror(
                "Visualization Error",
                f"An error occurred during visualization: {str(e)}",
            )

    def visualize_advanced_data(self) -> None:
        """
        Visualizes advanced analysis results as an interactive knowledge graph.
        """
        try:
            results = self.data_analysis_manager.fetch_all_analysis_results()
            if not results:
                tk.messagebox.showinfo(
                    "No Analysis Data", "No analysis data available to visualize."
                )
                return

            visualization_window = tk.Toplevel()
            visualization_window.title("Advanced Analysis Visualization")
            graph = self._create_knowledge_graph(results)
            fig, ax = plt.subplots(figsize=(8, 8))
            nx.draw_networkx(graph, nx.spring_layout(graph), ax=ax)
            canvas = FigureCanvasTkAgg(fig, master=visualization_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            fig.canvas.mpl_connect(
                "pick_event", lambda event: self._on_node_click(event, graph)
            )

        except Exception as e:
            logging.error(f"Error visualizing advanced analysis results: {e}")
            tk.messagebox.showerror(
                "Advanced Visualization Error",
                f"An error occurred during advanced visualization: {str(e)}",
            )

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalizes the PCA embeddings for plotting.

        :param embeddings: np.ndarray - The PCA embeddings to normalize.
        :return: np.ndarray - The normalized embeddings.
        """
        return (embeddings - np.min(embeddings, axis=0)) / (
            np.max(embeddings, axis=0) - np.min(embeddings, axis=0)
        )

    def _plot_embeddings(
        self, canvas: tk.Canvas, embeddings: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Plots the embeddings as points on the canvas with labels.

        :param canvas: tk.Canvas - The canvas to plot on.
        :param embeddings: np.ndarray - The normalized embeddings.
        :param labels: np.ndarray - The labels for each embedding.
        """
        for embedding, label in zip(embeddings, labels):
            x, y = embedding[0] * 700 + 50, embedding[1] * 700 + 50
            color = f"#{label % 256:02x}{(label * 3) % 256:02x}{(label * 7) % 256:02x}"
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color)
            canvas.create_text(
                x, y - 10, text=str(label), fill="black", font=("Arial", 8)
            )

    def _on_click(
        self, event: tk.Event, embeddings: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Handles click events on the visualization canvas, displaying information about the closest point.

        :param event: tk.Event - The click event.
        :param embeddings: np.ndarray - The normalized embeddings.
        :param labels: np.ndarray - The labels for each embedding.
        """
        x, y = event.x, event.y
        closest_point, min_distance = None, float("inf")
        for embedding, label in zip(embeddings, labels):
            embedding_x, embedding_y = embedding[0] * 700 + 50, embedding[1] * 700 + 50
            distance = ((embedding_x - x) ** 2 + (embedding_y - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance, closest_point = distance, (embedding, label)
        if closest_point:
            embedding, label = closest_point
            tk.messagebox.showinfo(
                "Embedding Info", f"Embedding: {embedding}\nLabel: {label}"
            )

    def _on_node_click(
        self, event: matplotlib.backend_bases.PickEvent, graph: nx.Graph
    ) -> None:
        """
        Handles node click events on the knowledge graph, displaying node information.

        :param event: matplotlib.backend_bases.PickEvent - The pick event on the graph.
        :param graph: nx.Graph - The knowledge graph.
        """
        node = event.artist.get_label()
        node_data = graph.nodes[node]
        tk.messagebox.showinfo(
            "Node Information", f"Node: {node}\nSize: {node_data['size']}"
        )

    def _create_knowledge_graph(self, results: List[Dict[str, Any]]) -> nx.Graph:
        """
        Creates a knowledge graph from analysis results.

        :param results: List[Dict[str, Any]] - The analysis results containing nodes and edges.
        :return: nx.Graph - The constructed knowledge graph.
        """
        graph = nx.Graph()
        for result in results:
            graph.add_node(result["label"], size=result["size"])
            for edge in result["edges"]:
                graph.add_edge(result["label"], edge["target"], weight=edge["weight"])
        return graph


class GUI:
    """
    Manages the application's main window and interactions, encapsulating all related functionalities.
    """

    def __init__(
        self,
        root: tk.Tk,
        db_manager: EmbeddingDatabaseManager,
        data_analysis_manager: DataAnalysisManager,
        model: SentenceTransformer,
        file_manager: FileManager,
        visualization_manager: DataVisualizationManager,
        advanced_visualization_manager: DataVisualizationManager,
    ):
        """
        Initializes the GUI with all necessary components and managers.
        """
        self.root = root
        self.db_manager = db_manager
        self.data_analysis_manager = data_analysis_manager
        self.model = model
        self.file_manager = file_manager
        self.visualization_manager = visualization_manager
        self.advanced_visualization_manager = advanced_visualization_manager
        self._initialize_user_interface()

    def _initialize_user_interface(self) -> None:
        """
        Configures the user interface, creating and organizing widgets.
        """
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.input_button = tk.Button(
            self.frame,
            text="Select Python Files Folder",
            command=self._handle_folder_selection,
        )
        self.input_button.pack(fill=tk.X)

        self.search_frame = tk.Frame(self.frame)
        self.search_frame.pack(fill=tk.X)

        self.search_entry = tk.Entry(self.search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.insert(0, "Enter search term...")

        self.search_button = tk.Button(
            self.search_frame,
            text="Search Embeddings",
            command=self._execute_embedding_search,
        )
        self.search_button.pack(side=tk.LEFT)

        self.results_text = tk.Text(self.frame, height=15, width=80)
        self.results_text.pack(fill=tk.X)

        self.analyze_button = tk.Button(
            self.frame,
            text="Analyze Embeddings",
            command=self._perform_embedding_analysis,
        )
        self.analyze_button.pack(fill=tk.X)

        self.visualize_button = tk.Button(
            self.frame,
            text="Visualize Embeddings",
            command=self._execute_embedding_visualization,
        )
        self.visualize_button.pack(fill=tk.X)

        self.advanced_visualize_button = tk.Button(
            self.frame,
            text="Visualize Analysis Data",
            command=self.advanced_visualization_manager.visualize_advanced_data,
        )
        self.advanced_visualize_button.pack(fill=tk.X)

    def _handle_folder_selection(self) -> None:
        """
        Manages folder selection and initiates processing of files within the selected folder.
        """
        folder_path = filedialog.askdirectory()
        if folder_path:
            total_files, processed_files = self.file_manager.process_folder(
                folder_path,
                lambda fp: self.file_manager.process_file(
                    fp, self.model, self.db_manager
                ),
            )
            self.db_manager.commit_changes()
            tk.messagebox.showinfo(
                "Processing Complete",
                f"All files have been processed. {processed_files}/{total_files} files processed.",
            )
        else:
            tk.messagebox.showinfo(
                "No folder selected", "Please select a valid folder."
            )

    def _execute_embedding_search(self) -> None:
        """
        Executes a search for embeddings based on the user's input and displays the results.
        """
        search_term = self.search_entry.get()
        results = self.db_manager.fetch_embeddings(search_term)
        self._present_search_results(results)

    def _present_search_results(
        self, results: List[Tuple[str, str, str, bytes]]
    ) -> None:
        """
        Presents search results in the text widget.
        """
        self.results_text.delete("1.0", tk.END)
        for imports, function, docstring, _ in results:
            result_text = f"Imports:\n{imports}\n\nFunction:\n{function}\n\nDocstring:\n{docstring}\n\n---\n\n"
            self.results_text.insert(tk.END, result_text)
        logging.info("Search results displayed.")

    def _perform_embedding_analysis(self) -> None:
        """
        Analyzes embeddings using various clustering and dimensionality reduction techniques.
        """
        try:
            results = self.db_manager.fetch_all_embeddings()
            embeddings = [
                np.frombuffer(result[3], dtype=np.float32) for result in results
            ]
            if embeddings:
                matrix = np.stack(embeddings)
                n_clusters = min(5, len(results))
                kmeans_labels, centroids = (
                    self.data_analysis_manager.perform_kmeans_clustering(
                        matrix, n_clusters
                    )
                )
                agglomerative_labels = (
                    self.data_analysis_manager.perform_agglomerative_clustering(
                        matrix, n_clusters
                    )
                )
                dbscan_labels = self.data_analysis_manager.perform_dbscan_clustering(
                    matrix, eps=0.5, min_samples=2
                )
                similarity_matrix = (
                    self.data_analysis_manager.perform_cosine_similarity(matrix)
                )
                pca_embeddings = self.data_analysis_manager.perform_pca(
                    matrix, n_components=2
                )
                tsne_embeddings = self.data_analysis_manager.perform_tsne(
                    matrix, n_components=2
                )
                pairwise_distances = (
                    self.data_analysis_manager.calculate_pairwise_distances(matrix)
                )
                kdtree = self.data_analysis_manager.build_kdtree(matrix)
                entropy_value = self.data_analysis_manager.calculate_entropy(
                    kmeans_labels
                )

                self.data_analysis_manager.store_analysis_results(
                    kmeans_labels,
                    centroids,
                    agglomerative_labels,
                    dbscan_labels,
                    similarity_matrix,
                    pca_embeddings,
                    tsne_embeddings,
                    pairwise_distances,
                    kdtree,
                    entropy_value,
                )

                messagebox.showinfo(
                    "Analysis Complete", "Embedding analysis completed successfully."
                )
            else:
                messagebox.showinfo("No Embeddings", "No embeddings found to analyze.")
        except Exception as e:
            logging.error(f"Error analyzing embeddings: {e}")
            messagebox.showerror(
                "Analysis Error", f"An error occurred during analysis: {str(e)}"
            )

    def _execute_embedding_visualization(self) -> None:
        """
        Visualizes embeddings using the provided visualization manager.
        """
        try:
            results = self.db_manager.fetch_all_embeddings()
            embeddings = [
                np.frombuffer(result[3], dtype=np.float32) for result in results
            ]
            if embeddings:
                matrix = np.stack(embeddings)
                kmeans_labels, _ = self.data_analysis_manager.perform_kmeans_clustering(
                    matrix, n_clusters=5
                )
                self.visualization_manager.visualize_basic_data(matrix, kmeans_labels)
            else:
                messagebox.showinfo(
                    "No Embeddings", "No embeddings found to visualize."
                )
        except Exception as e:
            logging.error(f"Error visualizing embeddings: {e}")
            messagebox.showerror(
                "Visualization Error",
                f"An error occurred during visualization: {str(e)}",
            )
