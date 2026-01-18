import asyncio
import enum
from functools import partial
import inspect
import logging
import traceback
from typing import Any, AsyncIterator, Generator, Generic, Optional, Tuple
import grpc
from grpc import _common
from grpc._cython import cygrpc
from . import _base_call
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadatumType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseType
from ._typing import SerializingFunction
class UnaryStreamCall(_StreamResponseMixin, Call, _base_call.UnaryStreamCall):
    """Object for managing unary-stream RPC calls.

    Returned when an instance of `UnaryStreamMultiCallable` object is called.
    """
    _request: RequestType
    _send_unary_request_task: asyncio.Task

    def __init__(self, request: RequestType, deadline: Optional[float], metadata: Metadata, credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], channel: cygrpc.AioChannel, method: bytes, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(channel.call(method, deadline, credentials, wait_for_ready), metadata, request_serializer, response_deserializer, loop)
        self._request = request
        self._context = cygrpc.build_census_context()
        self._send_unary_request_task = loop.create_task(self._send_unary_request())
        self._init_stream_response_mixin(self._send_unary_request_task)

    async def _send_unary_request(self) -> ResponseType:
        serialized_request = _common.serialize(self._request, self._request_serializer)
        try:
            await self._cython_call.initiate_unary_stream(serialized_request, self._metadata, self._context)
        except asyncio.CancelledError:
            if not self.cancelled():
                self.cancel()
            raise

    async def wait_for_connection(self) -> None:
        await self._send_unary_request_task
        if self.done():
            await self._raise_for_status()