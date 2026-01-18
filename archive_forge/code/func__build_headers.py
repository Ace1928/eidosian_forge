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
def _build_headers(self, options: FinalRequestOptions) -> httpx.Headers:
    custom_headers = options.headers or {}
    headers_dict = _merge_mappings(self.default_headers, custom_headers)
    self._validate_headers(headers_dict, custom_headers)
    headers = httpx.Headers(headers_dict)
    idempotency_header = self._idempotency_header
    if idempotency_header and options.method.lower() != 'get' and (idempotency_header not in headers):
        headers[idempotency_header] = options.idempotency_key or self._idempotency_key()
    return headers