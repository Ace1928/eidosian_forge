import asyncio
import inspect
from collections import deque
from typing import (
def accepts_context(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a context argument."""
    try:
        return inspect.signature(callable).parameters.get('context') is not None
    except ValueError:
        return False