from __future__ import annotations
import datetime
import logging
import os
import re
import socket
import sys
import typing
import warnings
from http.client import HTTPConnection as _HTTPConnection
from http.client import HTTPException as HTTPException  # noqa: F401
from http.client import ResponseNotReady
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .util.response import assert_header_parsing
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT, Timeout
from .util.util import to_str
from .util.wait import wait_for_read
from ._base_connection import _TYPE_BODY
from ._base_connection import ProxyConfig as ProxyConfig
from ._base_connection import _ResponseOptions as _ResponseOptions
from ._version import __version__
from .exceptions import (
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection, ssl_
from .util.request import body_to_chunks
from .util.ssl_ import assert_fingerprint as _assert_fingerprint
from .util.ssl_ import (
from .util.ssl_match_hostname import CertificateError, match_hostname
from .util.url import Url
class _WrappedAndVerifiedSocket(typing.NamedTuple):
    """
    Wrapped socket and whether the connection is
    verified after the TLS handshake
    """
    socket: ssl.SSLSocket | SSLTransport
    is_verified: bool