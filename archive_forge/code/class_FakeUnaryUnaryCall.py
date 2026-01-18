import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
class FakeUnaryUnaryCall(_WrappedUnaryUnaryCall):
    """Fake implementation for unary-unary RPCs.

    It is a dummy object for response message. Supply the intended response
    upon the initialization, and the coroutine will return the exact response
    message.
    """

    def __init__(self, response=object()):
        self.response = response
        self._future = asyncio.get_event_loop().create_future()
        self._future.set_result(self.response)

    def __await__(self):
        response = (yield from self._future.__await__())
        return response