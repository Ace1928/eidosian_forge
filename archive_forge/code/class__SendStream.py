import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar
class _SendStream(Generic[T]):

    def __init__(self, reader_loop: AbstractEventLoop, queue: Queue, done: object) -> None:
        """Create a writer for the queue and done object.

        Args:
            reader_loop: The event loop to use for the writer. This loop will be used
                         to schedule the writes to the queue.
            queue: The queue to write to. This is an asyncio queue.
            done: Special sentinel object to indicate that the writer is done.
        """
        self._reader_loop = reader_loop
        self._queue = queue
        self._done = done

    async def send(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop."""
        return self.send_nowait(item)

    def send_nowait(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop."""
        self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def aclose(self) -> None:
        """Schedule the done object write the queue using the original loop."""
        return self.close()

    def close(self) -> None:
        """Schedule the done object write the queue using the original loop."""
        self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, self._done)