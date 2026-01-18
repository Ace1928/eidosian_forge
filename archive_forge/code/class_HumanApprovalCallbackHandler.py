from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
class HumanApprovalCallbackHandler(BaseCallbackHandler):
    """Callback for manually validating values."""
    raise_error: bool = True

    def __init__(self, approve: Callable[[Any], bool]=_default_approve, should_check: Callable[[Dict[str, Any]], bool]=_default_true):
        self._approve = approve
        self._should_check = should_check

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        if self._should_check(serialized) and (not self._approve(input_str)):
            raise HumanRejectedException(f'Inputs {input_str} to tool {serialized} were rejected.')