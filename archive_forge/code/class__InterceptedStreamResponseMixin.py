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
class _InterceptedStreamResponseMixin:
    _response_aiter: Optional[AsyncIterable[ResponseType]]

    def _init_stream_response_mixin(self) -> None:
        self._response_aiter = None

    async def _wait_for_interceptor_task_response_iterator(self) -> ResponseType:
        call = await self._interceptors_task
        async for response in call:
            yield response

    def __aiter__(self) -> AsyncIterable[ResponseType]:
        if self._response_aiter is None:
            self._response_aiter = self._wait_for_interceptor_task_response_iterator()
        return self._response_aiter

    async def read(self) -> ResponseType:
        if self._response_aiter is None:
            self._response_aiter = self._wait_for_interceptor_task_response_iterator()
        return await self._response_aiter.asend(None)