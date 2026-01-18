from __future__ import annotations
import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from sqlalchemy import REAL, Column, String, Table, create_engine, insert, text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
Return connection string from database parameters.