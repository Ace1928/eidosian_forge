from abc import ABCMeta
from abc import abstractmethod
import asyncio
import collections
import functools
from typing import (
import grpc
from grpc._cython import cygrpc
from . import _base_call
from ._call import AioRpcError
from ._call import StreamStreamCall
from ._call import StreamUnaryCall
from ._call import UnaryStreamCall
from ._call import UnaryUnaryCall
from ._call import _API_STYLE_ERROR
from ._call import _RPC_ALREADY_FINISHED_DETAILS
from ._call import _RPC_HALF_CLOSED_DETAILS
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseIterableType
from ._typing import ResponseType
from ._typing import SerializingFunction
from ._utils import _timeout_to_deadline
class InterceptedStreamStreamCall(_InterceptedStreamResponseMixin, _InterceptedStreamRequestMixin, InterceptedCall, _base_call.StreamStreamCall):
    """Used for running a `StreamStreamCall` wrapped by interceptors."""
    _loop: asyncio.AbstractEventLoop
    _channel: cygrpc.AioChannel
    _last_returned_call_from_interceptors = Optional[_base_call.StreamStreamCall]

    def __init__(self, interceptors: Sequence[StreamStreamClientInterceptor], request_iterator: Optional[RequestIterableType], timeout: Optional[float], metadata: Metadata, credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], channel: cygrpc.AioChannel, method: bytes, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._channel = channel
        self._init_stream_response_mixin()
        request_iterator = self._init_stream_request_mixin(request_iterator)
        self._last_returned_call_from_interceptors = None
        interceptors_task = loop.create_task(self._invoke(interceptors, method, timeout, metadata, credentials, wait_for_ready, request_iterator, request_serializer, response_deserializer))
        super().__init__(interceptors_task)

    async def _invoke(self, interceptors: Sequence[StreamStreamClientInterceptor], method: bytes, timeout: Optional[float], metadata: Optional[Metadata], credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], request_iterator: RequestIterableType, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction) -> StreamStreamCall:
        """Run the RPC call wrapped in interceptors"""

        async def _run_interceptor(interceptors: List[StreamStreamClientInterceptor], client_call_details: ClientCallDetails, request_iterator: RequestIterableType) -> _base_call.StreamStreamCall:
            if interceptors:
                continuation = functools.partial(_run_interceptor, interceptors[1:])
                call_or_response_iterator = await interceptors[0].intercept_stream_stream(continuation, client_call_details, request_iterator)
                if isinstance(call_or_response_iterator, _base_call.StreamStreamCall):
                    self._last_returned_call_from_interceptors = call_or_response_iterator
                else:
                    self._last_returned_call_from_interceptors = StreamStreamCallResponseIterator(self._last_returned_call_from_interceptors, call_or_response_iterator)
                return self._last_returned_call_from_interceptors
            else:
                self._last_returned_call_from_interceptors = StreamStreamCall(request_iterator, _timeout_to_deadline(client_call_details.timeout), client_call_details.metadata, client_call_details.credentials, client_call_details.wait_for_ready, self._channel, client_call_details.method, request_serializer, response_deserializer, self._loop)
                return self._last_returned_call_from_interceptors
        client_call_details = ClientCallDetails(method, timeout, metadata, credentials, wait_for_ready)
        return await _run_interceptor(list(interceptors), client_call_details, request_iterator)

    def time_remaining(self) -> Optional[float]:
        raise NotImplementedError()