from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from tenacity import RetryCallState
class ToolManagerMixin:
    """Mixin for tool callbacks."""

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when tool errors."""