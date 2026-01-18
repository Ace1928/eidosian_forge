import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar
class _ReceiveStream(Generic[T]):

    def __init__(self, queue: Queue, done: object) -> None:
        """Create a reader for the queue and done object.

        This reader should be used in the same loop as the loop that was passed
        to the channel.
        """
        self._queue = queue
        self._done = done
        self._is_closed = False

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is self._done:
                self._is_closed = True
                break
            yield item