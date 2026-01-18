import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@staticmethod
def _es_vector_store(embedding: Optional[Embeddings]=None, **kwargs: Any) -> 'EcloudESVectorStore':
    index_name = kwargs.get('index_name')
    if index_name is None:
        raise ValueError('Please provide an index_name.')
    es_url = kwargs.get('es_url')
    if es_url is None:
        raise ValueError('Please provided a valid es connection url')
    return EcloudESVectorStore(embedding=embedding, **kwargs)