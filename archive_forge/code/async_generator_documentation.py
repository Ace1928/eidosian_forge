from __future__ import annotations
from asyncio import get_running_loop
from contextlib import asynccontextmanager
from queue import Empty, Full, Queue
from typing import Any, AsyncGenerator, Callable, Iterable, TypeVar
from .utils import run_in_executor_with_context

        Consume the generator in background thread.
        When items are received, they'll be pushed to the queue.
        