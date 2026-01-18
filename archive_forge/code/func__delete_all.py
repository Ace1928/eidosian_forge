from __future__ import annotations
import time
from itertools import repeat
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _delete_all(self) -> None:
    """Delete all records in the table."""
    while True:
        r = self._client.data().query(self._table_name, payload={'columns': ['id']})
        if r.status_code != 200:
            raise Exception(f'Error running query: {r.status_code} {r}')
        ids = [rec['id'] for rec in r['records']]
        if len(ids) == 0:
            break
        operations = [{'delete': {'table': self._table_name, 'id': id}} for id in ids]
        self._client.records().transaction(payload={'operations': operations})