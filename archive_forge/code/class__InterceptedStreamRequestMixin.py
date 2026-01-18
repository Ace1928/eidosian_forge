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
class _InterceptedStreamRequestMixin:
    _write_to_iterator_async_gen: Optional[AsyncIterable[RequestType]]
    _write_to_iterator_queue: Optional[asyncio.Queue]
    _status_code_task: Optional[asyncio.Task]
    _FINISH_ITERATOR_SENTINEL = object()

    def _init_stream_request_mixin(self, request_iterator: Optional[RequestIterableType]) -> RequestIterableType:
        if request_iterator is None:
            self._write_to_iterator_queue = asyncio.Queue(maxsize=1)
            self._write_to_iterator_async_gen = self._proxy_writes_as_request_iterator()
            self._status_code_task = None
            request_iterator = self._write_to_iterator_async_gen
        else:
            self._write_to_iterator_queue = None
        return request_iterator

    async def _proxy_writes_as_request_iterator(self):
        await self._interceptors_task
        while True:
            value = await self._write_to_iterator_queue.get()
            if value is _InterceptedStreamRequestMixin._FINISH_ITERATOR_SENTINEL:
                break
            yield value

    async def _write_to_iterator_queue_interruptible(self, request: RequestType, call: InterceptedCall):
        if self._status_code_task is None:
            self._status_code_task = self._loop.create_task(call.code())
        await asyncio.wait((self._loop.create_task(self._write_to_iterator_queue.put(request)), self._status_code_task), return_when=asyncio.FIRST_COMPLETED)

    async def write(self, request: RequestType) -> None:
        if self._write_to_iterator_queue is None:
            raise cygrpc.UsageError(_API_STYLE_ERROR)
        try:
            call = await self._interceptors_task
        except (asyncio.CancelledError, AioRpcError):
            raise asyncio.InvalidStateError(_RPC_ALREADY_FINISHED_DETAILS)
        if call.done():
            raise asyncio.InvalidStateError(_RPC_ALREADY_FINISHED_DETAILS)
        elif call._done_writing_flag:
            raise asyncio.InvalidStateError(_RPC_HALF_CLOSED_DETAILS)
        await self._write_to_iterator_queue_interruptible(request, call)
        if call.done():
            raise asyncio.InvalidStateError(_RPC_ALREADY_FINISHED_DETAILS)

    async def done_writing(self) -> None:
        """Signal peer that client is done writing.

        This method is idempotent.
        """
        if self._write_to_iterator_queue is None:
            raise cygrpc.UsageError(_API_STYLE_ERROR)
        try:
            call = await self._interceptors_task
        except asyncio.CancelledError:
            raise asyncio.InvalidStateError(_RPC_ALREADY_FINISHED_DETAILS)
        await self._write_to_iterator_queue_interruptible(_InterceptedStreamRequestMixin._FINISH_ITERATOR_SENTINEL, call)