from __future__ import annotations
import datetime
import os
from typing import (
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _create_weaviate_client(url: Optional[str]=None, api_key: Optional[str]=None, **kwargs: Any) -> weaviate.Client:
    try:
        import weaviate
    except ImportError:
        raise ImportError('Could not import weaviate python  package. Please install it with `pip install weaviate-client`')
    url = url or os.environ.get('WEAVIATE_URL')
    api_key = api_key or os.environ.get('WEAVIATE_API_KEY')
    auth = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth, **kwargs)