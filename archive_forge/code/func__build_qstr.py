from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
def _build_qstr(self, q_emb: List[float], topk: int, where_str: Optional[str]=None) -> str:
    q_emb_str = ','.join(map(str, q_emb))
    if where_str:
        where_str = f'PREWHERE {where_str}'
    else:
        where_str = ''
    q_str = f'\n            SELECT {self.config.column_map['text']}, dist, \n                {','.join(self.must_have_cols)}\n            FROM {self.config.database}.{self.config.table}\n            {where_str}\n            ORDER BY distance({self.config.column_map['vector']}, [{q_emb_str}]) \n                AS dist {self.dist_order}\n            LIMIT {topk}\n            '
    return q_str