from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
class DocumentDBSimilarityType(str, Enum):
    """DocumentDB Similarity Type as enumerator."""
    COS = 'cosine'
    'Cosine similarity'
    DOT = 'dotProduct'
    'Dot product'
    EUC = 'euclidean'
    'Euclidean distance'