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
def _warn_about_tls_in_tls(self, underlying_transport: asyncio.Transport, req: ClientRequest) -> None:
    """Issue a warning if the requested URL has HTTPS scheme."""
    if req.request_info.url.scheme != 'https':
        return
    asyncio_supports_tls_in_tls = getattr(underlying_transport, '_start_tls_compatible', False)
    if asyncio_supports_tls_in_tls:
        return
    warnings.warn("An HTTPS request is being sent through an HTTPS proxy. This support for TLS in TLS is known to be disabled in the stdlib asyncio (Python <3.11). This is why you'll probably see an error in the log below.\n\nIt is possible to enable it via monkeypatching. For more details, see:\n* https://bugs.python.org/issue37179\n* https://github.com/python/cpython/pull/28073\n\nYou can temporarily patch this as follows:\n* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n* https://github.com/aio-libs/aiohttp/discussions/6044\n", RuntimeWarning, source=self, stacklevel=3)