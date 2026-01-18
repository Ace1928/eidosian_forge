from __future__ import annotations
from typing import Any, Callable, List, Optional, Protocol, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
class CombineDocsProtocol(Protocol):
    """Interface for the combine_docs method."""

    def __call__(self, docs: List[Document], **kwargs: Any) -> str:
        """Interface for the combine_docs method."""