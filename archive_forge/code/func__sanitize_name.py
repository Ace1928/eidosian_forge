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
def _sanitize_name(input_str: str) -> str:
    return re.sub('[^a-zA-Z0-9_]', '', input_str)