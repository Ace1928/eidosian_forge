import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .helpers import (
from .http import (
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
@attr.s(auto_attribs=True, frozen=True, slots=True)
class RequestInfo:
    url: URL
    method: str
    headers: 'CIMultiDictProxy[str]'
    real_url: URL = attr.ib()

    @real_url.default
    def real_url_default(self) -> URL:
        return self.url