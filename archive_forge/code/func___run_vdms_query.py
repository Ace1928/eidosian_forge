from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def __run_vdms_query(self, all_queries: List[Dict], all_blobs: Optional[List]=[], print_last_response: Optional[bool]=False) -> Tuple[Any, Any]:
    response, response_array = self._client.query(all_queries, all_blobs)
    _ = _check_valid_response(all_queries, response)
    if print_last_response:
        self._client.print_last_response()
    return (response, response_array)