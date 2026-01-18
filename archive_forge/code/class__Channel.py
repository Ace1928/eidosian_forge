import collections
import sys
import types
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import grpc
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadataType
from ._typing import RequestIterableType
from ._typing import SerializingFunction
class _Channel(grpc.Channel):
    _channel: grpc.Channel
    _interceptor: Union[grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor, grpc.StreamStreamClientInterceptor, grpc.StreamUnaryClientInterceptor]

    def __init__(self, channel: grpc.Channel, interceptor: Union[grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor, grpc.StreamStreamClientInterceptor, grpc.StreamUnaryClientInterceptor]):
        self._channel = channel
        self._interceptor = interceptor

    def subscribe(self, callback: Callable, try_to_connect: Optional[bool]=False):
        self._channel.subscribe(callback, try_to_connect=try_to_connect)

    def unsubscribe(self, callback: Callable):
        self._channel.unsubscribe(callback)

    def unary_unary(self, method: str, request_serializer: Optional[SerializingFunction]=None, response_deserializer: Optional[DeserializingFunction]=None) -> grpc.UnaryUnaryMultiCallable:
        thunk = lambda m: self._channel.unary_unary(m, request_serializer, response_deserializer)
        if isinstance(self._interceptor, grpc.UnaryUnaryClientInterceptor):
            return _UnaryUnaryMultiCallable(thunk, method, self._interceptor)
        else:
            return thunk(method)

    def unary_stream(self, method: str, request_serializer: Optional[SerializingFunction]=None, response_deserializer: Optional[DeserializingFunction]=None) -> grpc.UnaryStreamMultiCallable:
        thunk = lambda m: self._channel.unary_stream(m, request_serializer, response_deserializer)
        if isinstance(self._interceptor, grpc.UnaryStreamClientInterceptor):
            return _UnaryStreamMultiCallable(thunk, method, self._interceptor)
        else:
            return thunk(method)

    def stream_unary(self, method: str, request_serializer: Optional[SerializingFunction]=None, response_deserializer: Optional[DeserializingFunction]=None) -> grpc.StreamUnaryMultiCallable:
        thunk = lambda m: self._channel.stream_unary(m, request_serializer, response_deserializer)
        if isinstance(self._interceptor, grpc.StreamUnaryClientInterceptor):
            return _StreamUnaryMultiCallable(thunk, method, self._interceptor)
        else:
            return thunk(method)

    def stream_stream(self, method: str, request_serializer: Optional[SerializingFunction]=None, response_deserializer: Optional[DeserializingFunction]=None) -> grpc.StreamStreamMultiCallable:
        thunk = lambda m: self._channel.stream_stream(m, request_serializer, response_deserializer)
        if isinstance(self._interceptor, grpc.StreamStreamClientInterceptor):
            return _StreamStreamMultiCallable(thunk, method, self._interceptor)
        else:
            return thunk(method)

    def _close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False

    def close(self):
        self._channel.close()