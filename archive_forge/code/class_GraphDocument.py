from __future__ import annotations
from typing import List, Union
from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Field
class GraphDocument(Serializable):
    """Represents a graph document consisting of nodes and relationships.

    Attributes:
        nodes (List[Node]): A list of nodes in the graph.
        relationships (List[Relationship]): A list of relationships in the graph.
        source (Document): The document from which the graph information is derived.
    """
    nodes: List[Node]
    relationships: List[Relationship]
    source: Document