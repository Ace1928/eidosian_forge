import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@staticmethod
def _bes_vector_store(embedding: Optional[Embeddings]=None, **kwargs: Any) -> 'BESVectorStore':
    index_name = kwargs.get('index_name')
    if index_name is None:
        raise ValueError('Please provide an index_name.')
    bes_url = kwargs.get('bes_url')
    if bes_url is None:
        raise ValueError('Please provided a valid bes connection url')
    return BESVectorStore(embedding=embedding, **kwargs)