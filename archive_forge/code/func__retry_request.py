from __future__ import annotations
import json
import time
import uuid
import email
import asyncio
import inspect
import logging
import platform
import warnings
import email.utils
from types import TracebackType
from random import random
from typing import (
from functools import lru_cache
from typing_extensions import Literal, override, get_origin
import anyio
import httpx
import distro
import pydantic
from httpx import URL, Limits
from pydantic import PrivateAttr
from . import _exceptions
from ._qs import Querystring
from ._files import to_httpx_files, async_to_httpx_files
from ._types import (
from ._utils import is_dict, is_list, is_given, is_mapping
from ._compat import model_copy, model_dump
from ._models import GenericModel, FinalRequestOptions, validate_type, construct_type
from ._response import (
from ._constants import (
from ._streaming import Stream, SSEDecoder, AsyncStream, SSEBytesDecoder
from ._exceptions import (
from ._legacy_response import LegacyAPIResponse
def _retry_request(self, options: FinalRequestOptions, cast_to: Type[ResponseT], remaining_retries: int, response_headers: httpx.Headers | None, *, stream: bool, stream_cls: type[_StreamT] | None) -> ResponseT | _StreamT:
    remaining = remaining_retries - 1
    if remaining == 1:
        log.debug('1 retry left')
    else:
        log.debug('%i retries left', remaining)
    timeout = self._calculate_retry_timeout(remaining, options, response_headers)
    log.info('Retrying request to %s in %f seconds', options.url, timeout)
    time.sleep(timeout)
    return self._request(options=options, cast_to=cast_to, remaining_retries=remaining, stream=stream, stream_cls=stream_cls)