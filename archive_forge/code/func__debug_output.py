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
def _debug_output(s: Any) -> None:
    """Print a debug message if DEBUG is True.

    Args:
        s: The message to print
    """
    if DEBUG:
        print(s)