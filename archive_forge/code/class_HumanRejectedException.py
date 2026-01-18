from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
class HumanRejectedException(Exception):
    """Exception to raise when a person manually review and rejects a value."""