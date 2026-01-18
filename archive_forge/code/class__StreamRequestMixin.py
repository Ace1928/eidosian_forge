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
class _StreamRequestMixin(Call):
    _metadata_sent: asyncio.Event
    _done_writing_flag: bool
    _async_request_poller: Optional[asyncio.Task]
    _request_style: _APIStyle

    def _init_stream_request_mixin(self, request_iterator: Optional[RequestIterableType]):
        self._metadata_sent = asyncio.Event()
        self._done_writing_flag = False
        if request_iterator is not None:
            self._async_request_poller = self._loop.create_task(self._consume_request_iterator(request_iterator))
            self._request_style = _APIStyle.ASYNC_GENERATOR
        else:
            self._async_request_poller = None
            self._request_style = _APIStyle.READER_WRITER

    def _raise_for_different_style(self, style: _APIStyle):
        if self._request_style is not style:
            raise cygrpc.UsageError(_API_STYLE_ERROR)

    def cancel(self) -> bool:
        if super().cancel():
            if self._async_request_poller is not None:
                self._async_request_poller.cancel()
            return True
        else:
            return False

    def _metadata_sent_observer(self):
        self._metadata_sent.set()

    async def _consume_request_iterator(self, request_iterator: RequestIterableType) -> None:
        try:
            if inspect.isasyncgen(request_iterator) or hasattr(request_iterator, '__aiter__'):
                async for request in request_iterator:
                    try:
                        await self._write(request)
                    except AioRpcError as rpc_error:
                        _LOGGER.debug('Exception while consuming the request_iterator: %s', rpc_error)
                        return
            else:
                for request in request_iterator:
                    try:
                        await self._write(request)
                    except AioRpcError as rpc_error:
                        _LOGGER.debug('Exception while consuming the request_iterator: %s', rpc_error)
                        return
            await self._done_writing()
        except:
            _LOGGER.debug('Client request_iterator raised exception:\n%s', traceback.format_exc())
            self.cancel()

    async def _write(self, request: RequestType) -> None:
        if self.done():
            raise asyncio.InvalidStateError(_RPC_ALREADY_FINISHED_DETAILS)
        if self._done_writing_flag:
            raise asyncio.InvalidStateError(_RPC_HALF_CLOSED_DETAILS)
        if not self._metadata_sent.is_set():
            await self._metadata_sent.wait()
            if self.done():
                await self._raise_for_status()
        serialized_request = _common.serialize(request, self._request_serializer)
        try:
            await self._cython_call.send_serialized_message(serialized_request)
        except cygrpc.InternalError as err:
            self._cython_call.set_internal_error(str(err))
            await self._raise_for_status()
        except asyncio.CancelledError:
            if not self.cancelled():
                self.cancel()
            raise

    async def _done_writing(self) -> None:
        if self.done():
            return
        if not self._done_writing_flag:
            self._done_writing_flag = True
            try:
                await self._cython_call.send_receive_close()
            except asyncio.CancelledError:
                if not self.cancelled():
                    self.cancel()
                raise

    async def write(self, request: RequestType) -> None:
        self._raise_for_different_style(_APIStyle.READER_WRITER)
        await self._write(request)

    async def done_writing(self) -> None:
        """Signal peer that client is done writing.

        This method is idempotent.
        """
        self._raise_for_different_style(_APIStyle.READER_WRITER)
        await self._done_writing()

    async def wait_for_connection(self) -> None:
        await self._metadata_sent.wait()
        if self.done():
            await self._raise_for_status()