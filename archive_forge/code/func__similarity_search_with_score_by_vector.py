from __future__ import annotations
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _similarity_search_with_score_by_vector(self, embedding: List[float], k: int=4, filter: Optional[str]=None, partition: str='default') -> List[Tuple[Document, float]]:
    """Return docs most similar to query vector, along with scores"""
    ret = self._collection.query(embedding, topk=k, filter=filter, partition=partition)
    if not ret:
        raise ValueError(f'Fail to query docs by vector, error {self._collection.message}')
    docs = []
    for doc in ret:
        metadata = doc.fields
        text = metadata.pop(self._text_field)
        score = doc.score
        docs.append((Document(page_content=text, metadata=metadata), score))
    return docs