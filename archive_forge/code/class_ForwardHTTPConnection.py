import logging
import ssl
from base64 import b64encode
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from .._backends.base import SOCKET_OPTION, NetworkBackend
from .._exceptions import ProxyError
from .._models import (
from .._ssl import default_ssl_context
from .._synchronization import Lock
from .._trace import Trace
from .connection import HTTPConnection
from .connection_pool import ConnectionPool
from .http11 import HTTP11Connection
from .interfaces import ConnectionInterface
class ForwardHTTPConnection(ConnectionInterface):

    def __init__(self, proxy_origin: Origin, remote_origin: Origin, proxy_headers: Union[HeadersAsMapping, HeadersAsSequence, None]=None, keepalive_expiry: Optional[float]=None, network_backend: Optional[NetworkBackend]=None, socket_options: Optional[Iterable[SOCKET_OPTION]]=None, proxy_ssl_context: Optional[ssl.SSLContext]=None) -> None:
        self._connection = HTTPConnection(origin=proxy_origin, keepalive_expiry=keepalive_expiry, network_backend=network_backend, socket_options=socket_options, ssl_context=proxy_ssl_context)
        self._proxy_origin = proxy_origin
        self._proxy_headers = enforce_headers(proxy_headers, name='proxy_headers')
        self._remote_origin = remote_origin

    def handle_request(self, request: Request) -> Response:
        headers = merge_headers(self._proxy_headers, request.headers)
        url = URL(scheme=self._proxy_origin.scheme, host=self._proxy_origin.host, port=self._proxy_origin.port, target=bytes(request.url))
        proxy_request = Request(method=request.method, url=url, headers=headers, content=request.stream, extensions=request.extensions)
        return self._connection.handle_request(proxy_request)

    def can_handle_request(self, origin: Origin) -> bool:
        return origin == self._remote_origin

    def close(self) -> None:
        self._connection.close()

    def info(self) -> str:
        return self._connection.info()

    def is_available(self) -> bool:
        return self._connection.is_available()

    def has_expired(self) -> bool:
        return self._connection.has_expired()

    def is_idle(self) -> bool:
        return self._connection.is_idle()

    def is_closed(self) -> bool:
        return self._connection.is_closed()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{self.info()}]>'