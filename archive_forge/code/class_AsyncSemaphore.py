import threading
from types import TracebackType
from typing import Optional, Type
from ._exceptions import ExceptionMapping, PoolTimeout, map_exceptions
class AsyncSemaphore:

    def __init__(self, bound: int) -> None:
        self._bound = bound
        self._backend = ''

    def setup(self) -> None:
        """
        Detect if we're running under 'asyncio' or 'trio' and create
        a semaphore with the correct implementation.
        """
        self._backend = current_async_library()
        if self._backend == 'trio':
            self._trio_semaphore = trio.Semaphore(initial_value=self._bound, max_value=self._bound)
        elif self._backend == 'asyncio':
            self._anyio_semaphore = anyio.Semaphore(initial_value=self._bound, max_value=self._bound)

    async def acquire(self) -> None:
        if not self._backend:
            self.setup()
        if self._backend == 'trio':
            await self._trio_semaphore.acquire()
        elif self._backend == 'asyncio':
            await self._anyio_semaphore.acquire()

    async def release(self) -> None:
        if self._backend == 'trio':
            self._trio_semaphore.release()
        elif self._backend == 'asyncio':
            self._anyio_semaphore.release()