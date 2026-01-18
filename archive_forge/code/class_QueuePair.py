import asyncio
import threading
from typing import List, Union, Any, TypeVar, Generic, Optional, Callable, Awaitable
from unittest.mock import AsyncMock
class QueuePair:
    called: asyncio.Queue
    results: asyncio.Queue

    def __init__(self):
        self.called = asyncio.Queue()
        self.results = asyncio.Queue()