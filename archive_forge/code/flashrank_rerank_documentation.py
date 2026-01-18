from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
Validate that api key and python package exists in environment.