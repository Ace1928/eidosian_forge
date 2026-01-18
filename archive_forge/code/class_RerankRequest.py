from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import Field
class RerankRequest:
    """Request for reranking."""

    def __init__(self, query: Any=None, passages: Any=None):
        self.query = query
        self.passages = passages if passages is not None else []