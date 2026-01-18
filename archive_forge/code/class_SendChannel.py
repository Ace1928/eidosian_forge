from __future__ import annotations
import socket
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar
import trio
class SendChannel(AsyncResource, Generic[SendType]):
    """A standard interface for sending Python objects to some receiver.

    `SendChannel` objects also implement the `AsyncResource` interface, so
    they can be closed by calling `~AsyncResource.aclose` or using an ``async
    with`` block.

    If you want to send raw bytes rather than Python objects, see
    `SendStream`.

    """
    __slots__ = ()

    @abstractmethod
    async def send(self, value: SendType) -> None:
        """Attempt to send an object through the channel, blocking if necessary.

        Args:
          value (object): The object to send.

        Raises:
          trio.BrokenResourceError: if something has gone wrong, and the
              channel is broken. For example, you may get this if the receiver
              has already been closed.
          trio.ClosedResourceError: if you previously closed this
              :class:`SendChannel` object, or if another task closes it while
              :meth:`send` is running.
          trio.BusyResourceError: some channels allow multiple tasks to call
              `send` at the same time, but others don't. If you try to call
              `send` simultaneously from multiple tasks on a channel that
              doesn't support it, then you can get `~trio.BusyResourceError`.

        """