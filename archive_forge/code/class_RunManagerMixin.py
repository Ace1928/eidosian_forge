from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class RunManagerMixin:
    """Mixin for run manager."""

    def on_text(self, text: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_retry(self, retry_state: RetryCallState, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run on a retry event."""