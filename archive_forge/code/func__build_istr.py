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
def _build_istr(self, transac: Iterable, column_names: Iterable[str]) -> str:
    ks = ','.join(column_names)
    _data = []
    for n in transac:
        n = ','.join([f"'{self.escape_str(str(_n))}'" for _n in n])
        _data.append(f'({n})')
    i_str = f'\n                INSERT INTO TABLE \n                    {self.config.database}.{self.config.table}({ks})\n                VALUES\n                {','.join(_data)}\n                '
    return i_str