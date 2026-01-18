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
class InterceptedCall:
    """Base implementation for all intercepted call arities.

    Interceptors might have some work to do before the RPC invocation with
    the capacity of changing the invocation parameters, and some work to do
    after the RPC invocation with the capacity for accessing to the wrapped
    `UnaryUnaryCall`.

    It handles also early and later cancellations, when the RPC has not even
    started and the execution is still held by the interceptors or when the
    RPC has finished but again the execution is still held by the interceptors.

    Once the RPC is finally executed, all methods are finally done against the
    intercepted call, being at the same time the same call returned to the
    interceptors.

    As a base class for all of the interceptors implements the logic around
    final status, metadata and cancellation.
    """
    _interceptors_task: asyncio.Task
    _pending_add_done_callbacks: Sequence[DoneCallbackType]

    def __init__(self, interceptors_task: asyncio.Task) -> None:
        self._interceptors_task = interceptors_task
        self._pending_add_done_callbacks = []
        self._interceptors_task.add_done_callback(self._fire_or_add_pending_done_callbacks)

    def __del__(self):
        self.cancel()

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

    def _wrap_add_done_callback(self, callback: DoneCallbackType, unused_call: _base_call.Call) -> None:
        callback(self)

    def cancel(self) -> bool:
        if not self._interceptors_task.done():
            return self._interceptors_task.cancel()
        try:
            call = self._interceptors_task.result()
        except AioRpcError:
            return False
        except asyncio.CancelledError:
            return False
        return call.cancel()

    def cancelled(self) -> bool:
        if not self._interceptors_task.done():
            return False
        try:
            call = self._interceptors_task.result()
        except AioRpcError as err:
            return err.code() == grpc.StatusCode.CANCELLED
        except asyncio.CancelledError:
            return True
        return call.cancelled()

    def done(self) -> bool:
        if not self._interceptors_task.done():
            return False
        try:
            call = self._interceptors_task.result()
        except (AioRpcError, asyncio.CancelledError):
            return True
        return call.done()

    def add_done_callback(self, callback: DoneCallbackType) -> None:
        if not self._interceptors_task.done():
            self._pending_add_done_callbacks.append(callback)
            return
        try:
            call = self._interceptors_task.result()
        except (AioRpcError, asyncio.CancelledError):
            callback(self)
            return
        if call.done():
            callback(self)
        else:
            callback = functools.partial(self._wrap_add_done_callback, callback)
            call.add_done_callback(callback)

    def time_remaining(self) -> Optional[float]:
        raise NotImplementedError()

    async def initial_metadata(self) -> Optional[Metadata]:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.initial_metadata()
        except asyncio.CancelledError:
            return None
        return await call.initial_metadata()

    async def trailing_metadata(self) -> Optional[Metadata]:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.trailing_metadata()
        except asyncio.CancelledError:
            return None
        return await call.trailing_metadata()

    async def code(self) -> grpc.StatusCode:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.code()
        except asyncio.CancelledError:
            return grpc.StatusCode.CANCELLED
        return await call.code()

    async def details(self) -> str:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.details()
        except asyncio.CancelledError:
            return _LOCAL_CANCELLATION_DETAILS
        return await call.details()

    async def debug_error_string(self) -> Optional[str]:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.debug_error_string()
        except asyncio.CancelledError:
            return ''
        return await call.debug_error_string()

    async def wait_for_connection(self) -> None:
        call = await self._interceptors_task
        return await call.wait_for_connection()