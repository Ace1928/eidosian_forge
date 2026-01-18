import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
@staticmethod
@functools.lru_cache(None)
def _make_ssl_context(verified: bool) -> SSLContext:
    if verified:
        return ssl.create_default_context()
    else:
        sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        sslcontext.options |= ssl.OP_NO_SSLv2
        sslcontext.options |= ssl.OP_NO_SSLv3
        sslcontext.check_hostname = False
        sslcontext.verify_mode = ssl.CERT_NONE
        try:
            sslcontext.options |= ssl.OP_NO_COMPRESSION
        except AttributeError as attr_err:
            warnings.warn('{!s}: The Python interpreter is compiled against OpenSSL < 1.0.0. Ref: https://docs.python.org/3/library/ssl.html#ssl.OP_NO_COMPRESSION'.format(attr_err))
        sslcontext.set_default_verify_paths()
        return sslcontext