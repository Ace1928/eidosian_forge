from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
def _default_true(_: Dict[str, Any]) -> bool:
    return True