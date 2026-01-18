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
def _merge_ssl_params(ssl: Union['SSLContext', bool, Fingerprint], verify_ssl: Optional[bool], ssl_context: Optional['SSLContext'], fingerprint: Optional[bytes]) -> Union['SSLContext', bool, Fingerprint]:
    if ssl is None:
        ssl = True
    if verify_ssl is not None and (not verify_ssl):
        warnings.warn('verify_ssl is deprecated, use ssl=False instead', DeprecationWarning, stacklevel=3)
        if ssl is not True:
            raise ValueError('verify_ssl, ssl_context, fingerprint and ssl parameters are mutually exclusive')
        else:
            ssl = False
    if ssl_context is not None:
        warnings.warn('ssl_context is deprecated, use ssl=context instead', DeprecationWarning, stacklevel=3)
        if ssl is not True:
            raise ValueError('verify_ssl, ssl_context, fingerprint and ssl parameters are mutually exclusive')
        else:
            ssl = ssl_context
    if fingerprint is not None:
        warnings.warn('fingerprint is deprecated, use ssl=Fingerprint(fingerprint) instead', DeprecationWarning, stacklevel=3)
        if ssl is not True:
            raise ValueError('verify_ssl, ssl_context, fingerprint and ssl parameters are mutually exclusive')
        else:
            ssl = Fingerprint(fingerprint)
    if not isinstance(ssl, SSL_ALLOWED_TYPES):
        raise TypeError('ssl should be SSLContext, bool, Fingerprint or None, got {!r} instead.'.format(ssl))
    return ssl