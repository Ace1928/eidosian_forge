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
def _fire_or_add_pending_done_callbacks(self, interceptors_task: asyncio.Task) -> None:
    if not self._pending_add_done_callbacks:
        return
    call_completed = False
    try:
        call = interceptors_task.result()
        if call.done():
            call_completed = True
    except (AioRpcError, asyncio.CancelledError):
        call_completed = True
    if call_completed:
        for callback in self._pending_add_done_callbacks:
            callback(self)
    else:
        for callback in self._pending_add_done_callbacks:
            callback = functools.partial(self._wrap_add_done_callback, callback)
            call.add_done_callback(callback)
    self._pending_add_done_callbacks = []