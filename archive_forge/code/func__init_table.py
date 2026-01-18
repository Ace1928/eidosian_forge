from __future__ import annotations
import uuid
from typing import Any, Iterable, List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _init_table(self) -> Any:
    import pyarrow as pa
    schema = pa.schema([pa.field(self._vector_key, pa.list_(pa.float32(), len(self.embeddings.embed_query('test')))), pa.field(self._id_key, pa.string()), pa.field(self._text_key, pa.string())])
    db = self.lancedb.connect('/tmp/lancedb')
    tbl = db.create_table(self._table_name, schema=schema, mode='overwrite')
    return tbl