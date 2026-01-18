from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _query_get(self, query_str: str, cache_name: str, local: bool=False) -> requests.Response:
    api_url = self._default_node + self._cache_url + '/' + cache_name + '?action=search&query=' + query_str + '&local=' + str(local)
    response = requests.get(api_url, timeout=REST_TIMEOUT)
    return response