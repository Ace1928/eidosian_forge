from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class CosmosDBVectorSearchType(str, Enum):
    """Cosmos DB Vector Search Type as enumerator."""
    VECTOR_IVF = 'vector-ivf'
    'IVF vector index'
    VECTOR_HNSW = 'vector-hnsw'
    'HNSW vector index'