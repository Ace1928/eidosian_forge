from __future__ import annotations
import importlib.util
import json
import re
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
def _sanitize_list_float(embedding: List[float]) -> List[float]:
    for value in embedding:
        if not isinstance(value, float):
            raise ValueError(f'Value ({value}) does not have type float')
    return embedding