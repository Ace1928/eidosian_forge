import asyncio
import threading
from typing import List, Union, Any, TypeVar, Generic, Optional, Callable, Awaitable
from unittest.mock import AsyncMock
class Box(Generic[T]):
    val: Optional[T]