from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _default_metadata(self, item: dict) -> dict:
    meta = dict(item)
    meta.pop(self._vectorfield, None)
    meta.pop(self._textfield, None)
    meta.pop('_type', None)
    return meta