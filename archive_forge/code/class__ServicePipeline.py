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
class _ServicePipeline(object):
    interceptors: Tuple[grpc.ServerInterceptor]

    def __init__(self, interceptors: Sequence[grpc.ServerInterceptor]):
        self.interceptors = tuple(interceptors)

    def _continuation(self, thunk: Callable, index: int) -> Callable:
        return lambda context: self._intercept_at(thunk, index, context)

    def _intercept_at(self, thunk: Callable, index: int, context: grpc.HandlerCallDetails) -> grpc.RpcMethodHandler:
        if index < len(self.interceptors):
            interceptor = self.interceptors[index]
            thunk = self._continuation(thunk, index + 1)
            return interceptor.intercept_service(thunk, context)
        else:
            return thunk(context)

    def execute(self, thunk: Callable, context: grpc.HandlerCallDetails) -> grpc.RpcMethodHandler:
        return self._intercept_at(thunk, 0, context)