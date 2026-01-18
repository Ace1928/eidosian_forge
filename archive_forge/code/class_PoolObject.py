from __future__ import annotations
import abc
import asyncio
import sqlite3
import contextlib
import functools
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence
from functools import partial, wraps
from typing import Any, Literal, Optional, overload, Awaitable, TypeVar
from anyio import CapacityLimiter, to_thread
class PoolObject(abc.ABC):
    """
    Base Pool Object
    """

    def __init__(self, *args, max_workers: Optional[int]=None, pool: Optional[ThreadPoolExecutor]=None, concurrent: Optional[bool]=False, **kwargs) -> None:
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=max_workers)
        self.pool = pool
        self.concurrent = concurrent

    def run_async(self, func: Callable[..., T], *args, **kwargs) -> Awaitable[T]:
        """
        Wraps the function in a thread pool
        """
        blocking = functools.partial(func, *args, **kwargs)
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self.pool, blocking)

    async def run_async(self, func: Callable[..., T], *args, **kwargs) -> Awaitable[T]:
        """
        Wraps the function in a thread pool
        """
        future = self.pool.submit(func, *args, **kwargs)
        f = asyncio.wrap_future(future)
        await f
        return f.result()