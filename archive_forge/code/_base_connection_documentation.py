from __future__ import annotations
import typing
from .util.connection import _TYPE_SOCKET_OPTIONS
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from .util.url import Url
Whether the connection has successfully connected to its proxy.
            This returns False if no proxy is in use. Used to determine whether
            errors are coming from the proxy layer or from tunnelling to the target origin.
            