from __future__ import annotations
import operator
from typing import Optional, Sequence
from langchain_community.cross_encoders import BaseCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Extra

        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        