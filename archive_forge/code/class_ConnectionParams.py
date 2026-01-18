from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class ConnectionParams:
    """Baidu VectorDB Connection params.

    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/s/6lrsob0wy

    Attribute:
        endpoint (str) : The access address of the vector database server
            that the client needs to connect to.
        api_key (str): API key for client to access the vector database server,
            which is used for authentication.
        account (str) : Account for client to access the vector database server.
        connection_timeout_in_mills (int) : Request Timeout.
    """

    def __init__(self, endpoint: str, api_key: str, account: str='root', connection_timeout_in_mills: int=50 * 1000):
        self.endpoint = endpoint
        self.api_key = api_key
        self.account = account
        self.connection_timeout_in_mills = connection_timeout_in_mills