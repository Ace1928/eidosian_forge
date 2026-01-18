from __future__ import annotations
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore
@xor_args(('query_texts', 'query_embeddings'))
def __query_cluster(self, query_texts: Optional[List[str]]=None, query_embeddings: Optional[List[List[float]]]=None, n_results: int=4, where: Optional[Dict[str, str]]=None, **kwargs: Any) -> List[Document]:
    """Query the BagelDB cluster based on the provided parameters."""
    try:
        import bagel
    except ImportError:
        raise ImportError('Please install bagel `pip install betabageldb`.')
    if self._embedding_function and query_embeddings is None and query_texts:
        texts = list(query_texts)
        query_embeddings = self._embedding_function.embed_documents(texts)
        query_texts = None
    return self._cluster.find(query_texts=query_texts, query_embeddings=query_embeddings, n_results=n_results, where=where, **kwargs)