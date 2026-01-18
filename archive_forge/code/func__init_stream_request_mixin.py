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
def _init_stream_request_mixin(self, request_iterator: Optional[RequestIterableType]):
    self._metadata_sent = asyncio.Event()
    self._done_writing_flag = False
    if request_iterator is not None:
        self._async_request_poller = self._loop.create_task(self._consume_request_iterator(request_iterator))
        self._request_style = _APIStyle.ASYNC_GENERATOR
    else:
        self._async_request_poller = None
        self._request_style = _APIStyle.READER_WRITER