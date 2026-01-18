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
class StreamStreamCallResponseIterator(_StreamCallResponseIterator, _base_call.StreamStreamCall):
    """StreamStreamCall class wich uses an alternative response iterator."""

    async def read(self) -> ResponseType:
        raise NotImplementedError()

    async def write(self, request: RequestType) -> None:
        raise NotImplementedError()

    async def done_writing(self) -> None:
        raise NotImplementedError()

    @property
    def _done_writing_flag(self) -> bool:
        return self._call._done_writing_flag