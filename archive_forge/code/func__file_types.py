from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_community.document_loaders.base_o365 import (
from langchain_community.document_loaders.parsers.registry import get_parser
@property
def _file_types(self) -> Sequence[_FileType]:
    """Return supported file types."""
    return (_FileType.DOC, _FileType.DOCX, _FileType.PDF)