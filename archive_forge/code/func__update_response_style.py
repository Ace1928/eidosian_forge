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
def _update_response_style(self, style: _APIStyle):
    if self._response_style is _APIStyle.UNKNOWN:
        self._response_style = style
    elif self._response_style is not style:
        raise cygrpc.UsageError(_API_STYLE_ERROR)