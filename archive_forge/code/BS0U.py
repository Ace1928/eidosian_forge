import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QListWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QLabel,
    QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from file_processor import FileProcessor
from embedding_storage import EmbeddingStorage
from data_analysis import DataAnalysis
from nlp_nlu_processor import NLPNLUProcessor
from knowledge_graph import KnowledgeGraph
from search_module import SearchModule


class GUI(QMainWindow):
    """
    Manages the graphical user interface, displaying the list of processed files and providing interactive functionalities.
    This implementation uses PyQt5 to create a robust, cross-platform GUI that integrates with all associated modules
    to provide a comprehensive, modern, and fully-featured interface.
    """

    def __init__(self, folder_path: str):
        """
        Initializes the GUI components and integrates with the backend modules.
        :param folder_path: str - Path to the folder containing documents to be processed and embedded.
        """
        super().__init__()
        self.setWindowTitle("Text and Embedding Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)  # Position and size: x, y, width, height

        self.file_processor = FileProcessor(folder_path)
        self.embedding_storage = EmbeddingStorage("embeddings.db", folder_path)
        self.data_analysis = DataAnalysis("analysis.db", "embeddings.db")
        self.nlp_processor = NLPNLUProcessor("embeddings.db", "nlp.db")
        self.knowledge_graph = KnowledgeGraph(
            "bolt://localhost:7687",
            "neo4j",
            "password",
            "embeddings.db",
            "analysis.db",
            "nlp.db",
            folder_path,
        )
        self.search_module = SearchModule(
            "search.db", "embeddings.db", "nlp.db", "analysis.db"
        )

        self.init_ui()

    def init_ui(self):
        """
        Sets up the user interface with tabs for each major functionality.
        """
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.file_tab = QWidget()
        self.embedding_tab = QWidget()
        self.analysis_tab = QWidget()
        self.search_tab = QWidget()
        self.knowledge_graph_tab = QWidget()

        self.tabs.addTab(self.file_tab, "File Processing")
        self.tabs.addTab(self.embedding_tab, "Embeddings")
        self.tabs.addTab(self.analysis_tab, "Data Analysis")
        self.tabs.addTab(self.search_tab, "Search")
        self.tabs.addTab(self.knowledge_graph_tab, "Knowledge Graph")

        self.setup_file_tab()
        self.setup_embedding_tab()
        self.setup_analysis_tab()
        self.setup_search_tab()
        self.setup_knowledge_graph_tab()

    def setup_file_tab(self):
        """
        Sets up the file processing tab with list of files and processing options.
        """
        layout = QVBoxLayout()
        self.file_list_widget = QListWidget()
        self.file_list_widget.addItems(self.file_processor._list_files())
        layout.addWidget(self.file_list_widget)

        process_button = QPushButton("Process Files")
        process_button.clicked.connect(self.process_files)
        layout.addWidget(process_button)

        self.file_tab.setLayout(layout)

    def setup_embedding_tab(self):
        """
        Sets up the embeddings tab to display and manage text embeddings.
        """
        layout = QVBoxLayout()
        self.embedding_list_widget = QListWidget()
        self.embedding_list_widget.addItems(
            [
                f"{file_path}"
                for file_path, _ in self.embedding_storage.load_embeddings()
            ]
        )
        layout.addWidget(self.embedding_list_widget)

        generate_button = QPushButton("Generate Embeddings")
        generate_button.clicked.connect(self.generate_embeddings)
        layout.addWidget(generate_button)

        self.embedding_tab.setLayout(layout)

    def setup_analysis_tab(self):
        """
        Sets up the data analysis tab to perform and display analysis results.
        """
        layout = QVBoxLayout()
        pca_button = QPushButton("Perform PCA")
        pca_button.clicked.connect(self.perform_pca)
        layout.addWidget(pca_button)

        tsne_button = QPushButton("Perform t-SNE")
        tsne_button.clicked.connect(self.perform_tsne)
        layout.addWidget(tsne_button)

        clustering_button = QPushButton("Perform Clustering")
        clustering_button.clicked.connect(self.perform_clustering)
        layout.addWidget(clustering_button)

        self.analysis_tab.setLayout(layout)

    def setup_search_tab(self):
        """
        Sets up the search tab to perform semantic searches.
        """
        layout = QVBoxLayout()
        self.search_query_input = QTextEdit()
        self.search_query_input.setPlaceholderText("Enter search query...")
        layout.addWidget(self.search_query_input)

        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_search)
        layout.addWidget(search_button)

        self.search_results_list = QListWidget()
        layout.addWidget(self.search_results_list)

        self.search_tab.setLayout(layout)

    def setup_knowledge_graph_tab(self):
        """
        Sets up the knowledge graph tab to visualize and manage the knowledge graph.
        """
        layout = QVBoxLayout()
        view_graph_button = QPushButton("View Knowledge Graph")
        view_graph_button.clicked.connect(self.view_knowledge_graph)
        layout.addWidget(view_graph_button)

        # This function needs to integrate with the backend to fetch the file's content and analysis results
        content = f"Contents and analysis for {file_path}"  # Placeholder for actual content fetching
        self.content_display.setText(content)
