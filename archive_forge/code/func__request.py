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
def _request(self, *, cast_to: Type[ResponseT], options: FinalRequestOptions, remaining_retries: int | None, stream: bool, stream_cls: type[_StreamT] | None) -> ResponseT | _StreamT:
    cast_to = self._maybe_override_cast_to(cast_to, options)
    self._prepare_options(options)
    retries = self._remaining_retries(remaining_retries, options)
    request = self._build_request(options)
    self._prepare_request(request)
    kwargs: HttpxSendArgs = {}
    if self.custom_auth is not None:
        kwargs['auth'] = self.custom_auth
    try:
        response = self._client.send(request, stream=stream or self._should_stream_response_body(request=request), **kwargs)
    except httpx.TimeoutException as err:
        log.debug('Encountered httpx.TimeoutException', exc_info=True)
        if retries > 0:
            return self._retry_request(options, cast_to, retries, stream=stream, stream_cls=stream_cls, response_headers=None)
        log.debug('Raising timeout error')
        raise APITimeoutError(request=request) from err
    except Exception as err:
        log.debug('Encountered Exception', exc_info=True)
        if retries > 0:
            return self._retry_request(options, cast_to, retries, stream=stream, stream_cls=stream_cls, response_headers=None)
        log.debug('Raising connection error')
        raise APIConnectionError(request=request) from err
    log.debug('HTTP Request: %s %s "%i %s"', request.method, request.url, response.status_code, response.reason_phrase)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        log.debug('Encountered httpx.HTTPStatusError', exc_info=True)
        if retries > 0 and self._should_retry(err.response):
            err.response.close()
            return self._retry_request(options, cast_to, retries, err.response.headers, stream=stream, stream_cls=stream_cls)
        if not err.response.is_closed:
            err.response.read()
        log.debug('Re-raising status error')
        raise self._make_status_error_from_response(err.response) from None
    return self._process_response(cast_to=cast_to, options=options, response=response, stream=stream, stream_cls=stream_cls)