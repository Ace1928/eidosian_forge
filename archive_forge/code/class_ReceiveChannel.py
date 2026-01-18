from __future__ import annotations
import socket
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar
import trio
class ReceiveChannel(AsyncResource, Generic[ReceiveType]):
    """A standard interface for receiving Python objects from some sender.

    You can iterate over a :class:`ReceiveChannel` using an ``async for``
    loop::

       async for value in receive_channel:
           ...

    This is equivalent to calling :meth:`receive` repeatedly. The loop exits
    without error when `receive` raises `~trio.EndOfChannel`.

    `ReceiveChannel` objects also implement the `AsyncResource` interface, so
    they can be closed by calling `~AsyncResource.aclose` or using an ``async
    with`` block.

    If you want to receive raw bytes rather than Python objects, see
    `ReceiveStream`.

    """
    __slots__ = ()

    @abstractmethod
    async def receive(self) -> ReceiveType:
        """Attempt to receive an incoming object, blocking if necessary.

        Returns:
          object: Whatever object was received.

        Raises:
          trio.EndOfChannel: if the sender has been closed cleanly, and no
              more objects are coming. This is not an error condition.
          trio.ClosedResourceError: if you previously closed this
              :class:`ReceiveChannel` object.
          trio.BrokenResourceError: if something has gone wrong, and the
              channel is broken.
          trio.BusyResourceError: some channels allow multiple tasks to call
              `receive` at the same time, but others don't. If you try to call
              `receive` simultaneously from multiple tasks on a channel that
              doesn't support it, then you can get `~trio.BusyResourceError`.

        """

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ReceiveType:
        try:
            return await self.receive()
        except trio.EndOfChannel:
            raise StopAsyncIteration from None