from abc import abstractmethod
from typing import Any, Dict, List
from langchain_community.graphs.graph_document import GraphDocument
@abstractmethod
def add_graph_documents(self, graph_documents: List[GraphDocument], include_source: bool=False) -> None:
    """Take GraphDocument as input as uses it to construct a graph."""
    pass