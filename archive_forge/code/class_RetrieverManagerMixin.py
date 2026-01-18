from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class RetrieverManagerMixin:
    """Mixin for Retriever callbacks."""

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when Retriever errors."""

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when Retriever ends running."""