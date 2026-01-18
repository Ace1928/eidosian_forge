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
def _make_status_error_from_response(self, response: httpx.Response) -> APIStatusError:
    if response.is_closed and (not response.is_stream_consumed):
        body = None
        err_msg = f'Error code: {response.status_code}'
    else:
        err_text = response.text.strip()
        body = err_text
        try:
            body = json.loads(err_text)
            err_msg = f'Error code: {response.status_code} - {body}'
        except Exception:
            err_msg = err_text or f'Error code: {response.status_code}'
    return self._make_status_error(err_msg, body=body, response=response)