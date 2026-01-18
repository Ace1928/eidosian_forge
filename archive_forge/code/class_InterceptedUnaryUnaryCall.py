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
class InterceptedUnaryUnaryCall(_InterceptedUnaryResponseMixin, InterceptedCall, _base_call.UnaryUnaryCall):
    """Used for running a `UnaryUnaryCall` wrapped by interceptors.

    For the `__await__` method is it is proxied to the intercepted call only when
    the interceptor task is finished.
    """
    _loop: asyncio.AbstractEventLoop
    _channel: cygrpc.AioChannel

    def __init__(self, interceptors: Sequence[UnaryUnaryClientInterceptor], request: RequestType, timeout: Optional[float], metadata: Metadata, credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], channel: cygrpc.AioChannel, method: bytes, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._channel = channel
        interceptors_task = loop.create_task(self._invoke(interceptors, method, timeout, metadata, credentials, wait_for_ready, request, request_serializer, response_deserializer))
        super().__init__(interceptors_task)

    async def _invoke(self, interceptors: Sequence[UnaryUnaryClientInterceptor], method: bytes, timeout: Optional[float], metadata: Optional[Metadata], credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], request: RequestType, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction) -> UnaryUnaryCall:
        """Run the RPC call wrapped in interceptors"""

        async def _run_interceptor(interceptors: List[UnaryUnaryClientInterceptor], client_call_details: ClientCallDetails, request: RequestType) -> _base_call.UnaryUnaryCall:
            if interceptors:
                continuation = functools.partial(_run_interceptor, interceptors[1:])
                call_or_response = await interceptors[0].intercept_unary_unary(continuation, client_call_details, request)
                if isinstance(call_or_response, _base_call.UnaryUnaryCall):
                    return call_or_response
                else:
                    return UnaryUnaryCallResponse(call_or_response)
            else:
                return UnaryUnaryCall(request, _timeout_to_deadline(client_call_details.timeout), client_call_details.metadata, client_call_details.credentials, client_call_details.wait_for_ready, self._channel, client_call_details.method, request_serializer, response_deserializer, self._loop)
        client_call_details = ClientCallDetails(method, timeout, metadata, credentials, wait_for_ready)
        return await _run_interceptor(list(interceptors), client_call_details, request)

    def time_remaining(self) -> Optional[float]:
        raise NotImplementedError()