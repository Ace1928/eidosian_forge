from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
def _similarity_search_without_score(self, embeddings: List[float], k: int=4, ef_search: int=40) -> List[Document]:
    """Returns a list of documents.

        Args:
            embeddings: The query vector
            k: the number of documents to return
            ef_search: Specifies the size of the dynamic candidate list
                that HNSW index uses during search. A higher value of
                efSearch provides better recall at cost of speed.

        Returns:
            A list of documents closest to the query vector
        """
    pipeline: List[dict[str, Any]] = [{'$search': {'vectorSearch': {'vector': embeddings, 'path': self._embedding_key, 'similarity': self._similarity_type, 'k': k, 'efSearch': ef_search}}}]
    cursor = self._collection.aggregate(pipeline)
    docs = []
    for res in cursor:
        text = res.pop(self._text_key)
        docs.append(Document(page_content=text, metadata=res))
    return docs