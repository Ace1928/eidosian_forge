import logging
import re
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchException
import json
from typing import List, Tuple, Dict, Any


class LogAnalyzer:
    """
    A class dedicated to analyzing log data using various text processing and machine learning techniques.

    Attributes:
        es_host (str): The hostname of the Elasticsearch instance.
        es_port (int): The port number of the Elasticsearch instance.
        index_name (str): The name of the Elasticsearch index to connect to.
        es (Elasticsearch): The Elasticsearch client instance.
        logger (logging.Logger): The logger instance for logging information, warnings, and errors.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer for converting log data into vector form.
        lemmatizer (WordNetLemmatizer): The lemmatizer for reducing words to their base form.
    """

    def __init__(self, es_host: str, es_port: int, index_name: str) -> None:
        """
        Initializes the LogAnalyzer with the specified Elasticsearch configuration and sets up text processing tools and logging.

        Args:
            es_host (str): The hostname of the Elasticsearch instance.
            es_port (int): The port number of the Elasticsearch instance.
            index_name (str): The name of the Elasticsearch index to connect to.
        """
        self.es_host = es_host
        self.es_port = es_port
        self.index_name = index_name
        self.es = None  # Elasticsearch client instance, to be initialized.
        self.logger = None  # Logger instance, to be initialized.
        self.connect_to_elasticsearch()
        self.configure_text_processing_tools()
        self.configure_logging()

    def connect_to_elasticsearch(self) -> None:
        """
        Establishes a connection to the Elasticsearch instance using the provided host and port.
        """
        try:
            self.es = Elasticsearch([{"host": self.es_host, "port": self.es_port}])
            self.logger.info("Successfully connected to Elasticsearch.")
        except ElasticsearchException as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def configure_text_processing_tools(self) -> None:
        """
        Configures the text processing tools used for analyzing log data.
        """
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def configure_logging(self) -> None:
        """
        Configures the logging mechanism for the LogAnalyzer.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(ch)

    def analyze_log(self, log_data: str) -> None:
        """
        Analyzes the provided log data, performing preprocessing, tokenization, vectorization, clustering, topic modeling, anomaly detection, and log correlation.

        Args:
            log_data (str): The raw log data to be analyzed.
        """
        try:
            log_text = self.preprocess_log(log_data)
            tokens = self.tokenize_log(log_text)
            vectors = self.vectorizer.fit_transform([" ".join(tokens)])
            clusters = self.cluster_logs(vectors)
            topics = self.model_topics(vectors)
            anomalies = self.detect_anomalies(topics)
            correlations = self.correlate_logs(clusters)
            self.visualize_logs(correlations)
        except Exception as e:
            self.logger.error(f"Error analyzing log: {e}")
            raise

    def preprocess_log(self, log_data: str) -> str:
        """
        Preprocesses the log data by removing special characters and extra spaces.

        Args:
            log_data (str): The raw log data to be preprocessed.

        Returns:
            str: The preprocessed log data.
        """
        log_text = re.sub(r"[\r\n\t]", "", log_data)
        log_text = re.sub(r"\s+", " ", log_text)
        self.logger.debug(f"Preprocessed log: {log_text}")
        return log_text

    def tokenize_log(self, log_text: str) -> List[str]:
        """
        Tokenizes the preprocessed log data into individual words and lemmatizes them.

        Args:
            log_text (str): The preprocessed log data to be tokenized.

        Returns:
            List[str]: The list of lemmatized tokens.
        """
        tokens = [self.lemmatizer.lemmatize(word) for word in log_text.split()]
        self.logger.debug(f"Tokenized log: {tokens}")
        return tokens

    def cluster_logs(self, vectors: Any) -> List[int]:
        """
        Clusters the log data into groups based on their TF-IDF vectors.

        Args:
            vectors (Any): The TF-IDF vectors of the log data.

        Returns:
            List[int]: The cluster labels for each log entry.
        """
        kmeans = KMeans(n_clusters=5)
        clusters = kmeans.fit_predict(vectors)
        self.logger.debug(f"Log clusters: {clusters}")
        return clusters

    def model_topics(self, vectors: Any) -> List[float]:
        """
        Models the topics present in the log data using Latent Dirichlet Allocation.

        Args:
            vectors (Any): The TF-IDF vectors of the log data.

        Returns:
            List[float]: The topic distribution for each log entry.
        """
        lda = LatentDirichletAllocation(n_components=5)
        topics = lda.fit_transform(vectors)
        self.logger.debug(f"Log topics: {topics}")
        return topics

    def detect_anomalies(self, topics: List[float]) -> List[Tuple[int, int]]:
        """
        Detects anomalies in the topic distribution by comparing the cosine similarity between topics.

        Args:
            topics (List[float]): The topic distribution for each log entry.

        Returns:
            List[Tuple[int, int]]: The pairs of log entries identified as anomalies.
        """
        anomalies = []
        for i, topic in enumerate(topics):
            for j, other_topic in enumerate(topics):
                if i != j:
                    similarity = cosine_similarity([topic], [other_topic])
                    if similarity < 0.5:
                        anomalies.append((i, j))
                        self.logger.debug(
                            f"Anomaly detected between topics {i} and {j}"
                        )
        return anomalies

    def correlate_logs(self, clusters: List[int]) -> Dict[int, List[int]]:
        """
        Correlates the logs based on their cluster assignments, identifying logs that belong to the same cluster.

        Args:
            clusters (List[int]): The cluster labels for each log entry.

        Returns:
            Dict[int, List[int]]: The correlations between log entries based on cluster assignments.
        """
        correlations = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i] == clusters[j]:
                    correlations.setdefault(i, []).append(j)
                    correlations.setdefault(j, []).append(i)
        self.logger.debug(f"Log correlations: {correlations}")
        return correlations

    def visualize_logs(self, correlations: Dict[int, List[int]]) -> None:
        """
        Visualizes the correlations between logs using a graph representation.

        Args:
            correlations (Dict[int, List[int]]): The correlations between log entries to be visualized.
        """
        G = nx.Graph()
        for key, values in correlations.items():
            for value in values:
                G.add_edge(key, value)
        plt.figure(figsize=(8, 6))
        nx.draw(
            G,
            with_labels=True,
            node_color="lightblue",
            font_weight="bold",
            node_size=700,
            font_size=10,
        )
        plt.title("Log Correlation Graph")
        plt.show()


if __name__ == "__main__":
    log_analyzer = LogAnalyzer("localhost", 9200, "log_index")
    log_data = json.dumps(
        {
            "message": "Sample log message",
            "level": "INFO",
            "timestamp": "2023-02-20T14:30:00Z",
        }
    )
    log_analyzer.analyze_log(log_data)
