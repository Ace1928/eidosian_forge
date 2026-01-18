from __future__ import annotations
import logging
import os
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from packaging import version
from langchain_community.vectorstores.utils import (
def _is_pinecone_v3() -> bool:
    pinecone = _import_pinecone()
    pinecone_client_version = pinecone.__version__
    return version.parse(pinecone_client_version) >= version.parse('3.0.0.dev')