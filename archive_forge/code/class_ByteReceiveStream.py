from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class ByteReceiveStream(AsyncResource, TypedAttributeProvider):
    """
    An interface for receiving bytes from a single peer.

    Iterating this byte stream will yield a byte string of arbitrary length, but no more
    than 65536 bytes.
    """

    def __aiter__(self) -> ByteReceiveStream:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.receive()
        except EndOfStream:
            raise StopAsyncIteration

    @abstractmethod
    async def receive(self, max_bytes: int=65536) -> bytes:
        """
        Receive at most ``max_bytes`` bytes from the peer.

        .. note:: Implementors of this interface should not return an empty
            :class:`bytes` object, and users should ignore them.

        :param max_bytes: maximum number of bytes to receive
        :return: the received bytes
        :raises ~anyio.EndOfStream: if this stream has been closed from the other end
        """