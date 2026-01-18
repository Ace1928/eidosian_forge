from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
def _messages_search_result_to_doc(self, results: List[MemorySearchResult]) -> List[Document]:
    return [Document(page_content=r.message.pop('content'), metadata={'score': r.dist, **r.message}) for r in results if r.message]