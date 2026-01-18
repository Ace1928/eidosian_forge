from __future__ import annotations
import logging
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _similarity_search_with_score(self, embedding: List[float], k: int=4, pre_filter: Optional[Dict]=None, post_filter_pipeline: Optional[List[Dict]]=None) -> List[Tuple[Document, float]]:
    params = {'queryVector': embedding, 'path': self._embedding_key, 'numCandidates': k * 10, 'limit': k, 'index': self._index_name}
    if pre_filter:
        params['filter'] = pre_filter
    query = {'$vectorSearch': params}
    pipeline = [query, {'$set': {'score': {'$meta': 'vectorSearchScore'}}}]
    if post_filter_pipeline is not None:
        pipeline.extend(post_filter_pipeline)
    cursor = self._collection.aggregate(pipeline)
    docs = []
    for res in cursor:
        text = res.pop(self._text_key)
        score = res.pop('score')
        docs.append((Document(page_content=text, metadata=res), score))
    return docs