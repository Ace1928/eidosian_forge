from __future__ import annotations
import json
import uuid
from typing import Any, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
def _ensure_table(self) -> None:
    """Ensures the table for storing embeddings exists."""
    create_table_sql = f'\n        CREATE TABLE IF NOT EXISTS {self._table_name} (\n            {self._id_key} VARCHAR PRIMARY KEY,\n            {self._text_key} VARCHAR,\n            {self._vector_key} FLOAT[],\n            metadata VARCHAR\n        )\n        '
    self._connection.execute(create_table_sql)