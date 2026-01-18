from __future__ import annotations
import errno
import logging
import queue
import sys
import typing
import warnings
import weakref
from socket import timeout as SocketTimeout
from types import TracebackType
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from ._request_methods import RequestMethods
from .connection import (
from .connection import port_by_scheme as port_by_scheme
from .exceptions import (
from .response import BaseHTTPResponse
from .util.connection import is_connection_dropped
from .util.proxy import connection_requires_http_tunnel
from .util.request import _TYPE_BODY_POSITION, set_file_position
from .util.retry import Retry
from .util.ssl_match_hostname import CertificateError
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_DEFAULT, Timeout
from .util.url import Url, _encode_target
from .util.url import _normalize_host as normalize_host
from .util.url import parse_url
from .util.util import to_str
def _url_from_pool(pool: HTTPConnectionPool | HTTPSConnectionPool, path: str | None=None) -> str:
    """Returns the URL from a given connection pool. This is mainly used for testing and logging."""
    return Url(scheme=pool.scheme, host=pool.host, port=pool.port, path=path).url