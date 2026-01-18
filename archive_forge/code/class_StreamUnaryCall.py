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
class StreamUnaryCall(_StreamRequestMixin, _UnaryResponseMixin, Call, _base_call.StreamUnaryCall):
    """Object for managing stream-unary RPC calls.

    Returned when an instance of `StreamUnaryMultiCallable` object is called.
    """

    def __init__(self, request_iterator: Optional[RequestIterableType], deadline: Optional[float], metadata: Metadata, credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], channel: cygrpc.AioChannel, method: bytes, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(channel.call(method, deadline, credentials, wait_for_ready), metadata, request_serializer, response_deserializer, loop)
        self._context = cygrpc.build_census_context()
        self._init_stream_request_mixin(request_iterator)
        self._init_unary_response_mixin(loop.create_task(self._conduct_rpc()))

    async def _conduct_rpc(self) -> ResponseType:
        try:
            serialized_response = await self._cython_call.stream_unary(self._metadata, self._metadata_sent_observer, self._context)
        except asyncio.CancelledError:
            if not self.cancelled():
                self.cancel()
            raise
        if self._cython_call.is_ok():
            return _common.deserialize(serialized_response, self._response_deserializer)
        else:
            return cygrpc.EOF