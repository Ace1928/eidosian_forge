from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union
from .._core._exceptions import EndOfStream
from .._core._typedattr import TypedAttributeProvider
from ._resources import AsyncResource
from ._tasks import TaskGroup
class UnreliableObjectReceiveStream(Generic[T_co], AsyncResource, TypedAttributeProvider):
    """
    An interface for receiving objects.

    This interface makes no guarantees that the received messages arrive in the order in
    which they were sent, or that no messages are missed.

    Asynchronously iterating over objects of this type will yield objects matching the
    given type parameter.
    """

    def __aiter__(self) -> UnreliableObjectReceiveStream[T_co]:
        return self

    async def __anext__(self) -> T_co:
        try:
            return await self.receive()
        except EndOfStream:
            raise StopAsyncIteration

    @abstractmethod
    async def receive(self) -> T_co:
        """
        Receive the next item.

        :raises ~anyio.ClosedResourceError: if the receive stream has been explicitly
            closed
        :raises ~anyio.EndOfStream: if this stream has been closed from the other end
        :raises ~anyio.BrokenResourceError: if this stream has been rendered unusable
            due to external causes
        """