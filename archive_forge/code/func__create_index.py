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
def _create_index(self):
    from google.api_core.exceptions import ClientError
    table = self.bq_client.get_table(self.vectors_table)
    if (table.num_rows or 0) < _MIN_INDEX_ROWS:
        return
    if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        distance_type = 'EUCLIDEAN'
    elif self.distance_strategy == DistanceStrategy.COSINE:
        distance_type = 'COSINE'
    else:
        distance_type = 'EUCLIDEAN'
    index_name = f'{self.table_name}_langchain_index'
    try:
        sql = f'\n                CREATE VECTOR INDEX IF NOT EXISTS\n                `{index_name}`\n                ON `{self.full_table_id}`({self.text_embedding_field})\n                OPTIONS(distance_type="{distance_type}", index_type="IVF")\n            '
        self.bq_client.query(sql).result()
        self._have_index = True
    except ClientError as ex:
        self._logger.debug('Vector index creation failed (%s).', ex.args[0])
    finally:
        self._creating_index = False