from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def _initialize_vector_index(self) -> Any:
    """
        A vector index in BigQuery table enables efficient
        approximate vector search.
        """
    from google.cloud import bigquery
    if self._have_index or self._creating_index:
        return
    table = self.bq_client.get_table(self.vectors_table)
    if (table.num_rows or 0) < _MIN_INDEX_ROWS:
        self._logger.debug('Not enough rows to create a vector index.')
        return
    if (datetime.utcnow() - self._last_index_check).total_seconds() < _INDEX_CHECK_PERIOD_SECONDS:
        return
    with _vector_table_lock:
        if self._creating_index or self._have_index:
            return
        self._last_index_check = datetime.utcnow()
        check_query = f"SELECT 1 FROM `{self.project_id}.{self.dataset_name}.INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE table_name = '{self.table_name}'"
        job = self.bq_client.query(check_query, api_method=bigquery.enums.QueryApiMethod.QUERY)
        if job.result().total_rows == 0:
            self._create_index_in_background()
        else:
            self._logger.debug('Vector index already exists.')
            self._have_index = True