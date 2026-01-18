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
def _should_retry(self, response: httpx.Response) -> bool:
    should_retry_header = response.headers.get('x-should-retry')
    if should_retry_header == 'true':
        log.debug('Retrying as header `x-should-retry` is set to `true`')
        return True
    if should_retry_header == 'false':
        log.debug('Not retrying as header `x-should-retry` is set to `false`')
        return False
    if response.status_code == 408:
        log.debug('Retrying due to status code %i', response.status_code)
        return True
    if response.status_code == 409:
        log.debug('Retrying due to status code %i', response.status_code)
        return True
    if response.status_code == 429:
        log.debug('Retrying due to status code %i', response.status_code)
        return True
    if response.status_code >= 500:
        log.debug('Retrying due to status code %i', response.status_code)
        return True
    log.debug('Not retrying')
    return False