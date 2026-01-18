from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_pipeline_vector_ivf(self, embeddings: List[float], k: int=4) -> List[dict[str, Any]]:
    pipeline: List[dict[str, Any]] = [{'$search': {'cosmosSearch': {'vector': embeddings, 'path': self._embedding_key, 'k': k}, 'returnStoredSource': True}}, {'$project': {'similarityScore': {'$meta': 'searchScore'}, 'document': '$$ROOT'}}]
    return pipeline