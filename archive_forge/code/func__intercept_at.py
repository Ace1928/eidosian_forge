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
def _intercept_at(self, thunk: Callable, index: int, context: grpc.HandlerCallDetails) -> grpc.RpcMethodHandler:
    if index < len(self.interceptors):
        interceptor = self.interceptors[index]
        thunk = self._continuation(thunk, index + 1)
        return interceptor.intercept_service(thunk, context)
    else:
        return thunk(context)