import collections
from typing import (
import duet.futuretools as futuretools
class AsyncCollector(Generic[T]):
    """Allows async iteration over values dynamically added by the client.

    This class is useful for creating an asynchronous iterator that is "fed" by
    one process (the "producer") and iterated over by another process (the
    "consumer"). The producer calls `.add` repeatedly to add values to be
    iterated over, and then calls either `.done` or `.error` to stop the
    iteration or raise an error, respectively. The consumer can use `async for`
    or direct calls to `__anext__` to iterate over the produced values.
    """

    def __init__(self):
        self._buffer: Deque[T] = collections.deque()
        self._waiter: Optional[futuretools.AwaitableFuture[None]] = None
        self._done: bool = False
        self._error: Optional[Exception] = None

    def add(self, value: T) -> None:
        if self._done:
            raise RuntimeError('already done.')
        self._buffer.append(value)
        if self._waiter:
            self._waiter.try_set_result(None)

    def done(self) -> None:
        if self._done:
            raise RuntimeError('already done.')
        self._done = True
        if self._waiter:
            self._waiter.try_set_result(None)

    def error(self, error: Exception) -> None:
        if self._done:
            raise RuntimeError('already done.')
        self._done = True
        self._error = error
        if self._waiter:
            self._waiter.try_set_result(None)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if not self._done and (not self._buffer):
            self._waiter = futuretools.AwaitableFuture()
            await self._waiter
            self._waiter = None
        if self._buffer:
            return self._buffer.popleft()
        if self._error:
            raise self._error
        raise StopAsyncIteration()