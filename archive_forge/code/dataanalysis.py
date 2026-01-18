class AnalysisDatabaseManager(DatabaseManager):
    """
    Manages the database operations specifically tailored for storing and retrieving data analysis results.
    This class ensures the initialization of the 'analysis' table, storage of analysis results, and retrieval of these results.
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

    def store_results(
        self, analysis_type: str, parameters: str, results: bytes
    ) -> None:
        """
        Store the results of a specific analysis in the database.

        Parameters:
        - analysis_type: str - The type of analysis performed.
        - parameters: str - The parameters used for the analysis.
        - results: bytes - The binary results of the analysis.
        """
        insert_query = (
            "INSERT INTO analysis (analysis_type, parameters, results) VALUES (?, ?, ?)"
        )
        try:
            self.cursor.execute(insert_query, (analysis_type, parameters, results))
        except sqlite3.DatabaseError as e:
            logging.error(f"Error storing analysis results: {e}")
            raise

    def retrieve_results(self, analysis_type: str, parameters: str) -> bytes | None:
        """
        Retrieve the results of a specific analysis from the database based on the analysis type and parameters.

        Parameters:
        - analysis_type: str - The type of analysis to retrieve.
        - parameters: str - The parameters used for the analysis.
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
