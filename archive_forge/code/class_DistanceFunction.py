from __future__ import annotations
import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class DistanceFunction(Enum):
    COSINE_SIM = 'COSINE_SIM'
    EUCLIDEAN_DIST = 'EUCLIDEAN_DIST'
    DOT_PRODUCT = 'DOT_PRODUCT'

    def order_by(self) -> str:
        if self.value == 'EUCLIDEAN_DIST':
            return 'ASC'
        return 'DESC'