from __future__ import annotations
import logging
from enum import Enum
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_pipeline_vector_hnsw(self, embeddings: List[float], k: int=4, ef_search: int=40) -> List[dict[str, Any]]:
    pipeline: List[dict[str, Any]] = [{'$search': {'cosmosSearch': {'vector': embeddings, 'path': self._embedding_key, 'k': k, 'efSearch': ef_search}}}, {'$project': {'similarityScore': {'$meta': 'searchScore'}, 'document': '$$ROOT'}}]
    return pipeline