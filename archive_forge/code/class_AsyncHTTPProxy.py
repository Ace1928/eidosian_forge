import logging
import ssl
from base64 import b64encode
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ProxyError
from .._models import (
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .connection import AsyncHTTPConnection
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface
class AsyncHTTPProxy(AsyncConnectionPool):
    """
    A connection pool that sends requests via an HTTP proxy.
    """

    def __init__(self, proxy_url: Union[URL, bytes, str], proxy_auth: Optional[Tuple[Union[bytes, str], Union[bytes, str]]]=None, proxy_headers: Union[HeadersAsMapping, HeadersAsSequence, None]=None, ssl_context: Optional[ssl.SSLContext]=None, proxy_ssl_context: Optional[ssl.SSLContext]=None, max_connections: Optional[int]=10, max_keepalive_connections: Optional[int]=None, keepalive_expiry: Optional[float]=None, http1: bool=True, http2: bool=False, retries: int=0, local_address: Optional[str]=None, uds: Optional[str]=None, network_backend: Optional[AsyncNetworkBackend]=None, socket_options: Optional[Iterable[SOCKET_OPTION]]=None) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            proxy_url: The URL to use when connecting to the proxy server.
                For example `"http://127.0.0.1:8080/"`.
            proxy_auth: Any proxy authentication as a two-tuple of
                (username, password). May be either bytes or ascii-only str.
            proxy_headers: Any HTTP headers to use for the proxy requests.
                For example `{"Proxy-Authorization": "Basic <username>:<password>"}`.
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore.default_ssl_context()`
                will be used.
            proxy_ssl_context: The same as `ssl_context`, but for a proxy server rather than a remote origin.
            max_connections: The maximum number of concurrent HTTP connections that
                the pool should allow. Any attempt to send a request on a pool that
                would exceed this amount will block until a connection is available.
            max_keepalive_connections: The maximum number of idle HTTP connections
                that will be maintained in the pool.
            keepalive_expiry: The duration in seconds that an idle HTTP connection
                may be maintained for before being expired from the pool.
            http1: A boolean indicating if HTTP/1.1 requests should be supported
                by the connection pool. Defaults to True.
            http2: A boolean indicating if HTTP/2 requests should be supported by
                the connection pool. Defaults to False.
            retries: The maximum number of retries when trying to establish
                a connection.
            local_address: Local address to connect from. Can also be used to
                connect using a particular address family. Using
                `local_address="0.0.0.0"` will connect using an `AF_INET` address
                (IPv4), while using `local_address="::"` will connect using an
                `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
        """
        super().__init__(ssl_context=ssl_context, max_connections=max_connections, max_keepalive_connections=max_keepalive_connections, keepalive_expiry=keepalive_expiry, http1=http1, http2=http2, network_backend=network_backend, retries=retries, local_address=local_address, uds=uds, socket_options=socket_options)
        self._proxy_url = enforce_url(proxy_url, name='proxy_url')
        if self._proxy_url.scheme == b'http' and proxy_ssl_context is not None:
            raise RuntimeError('The `proxy_ssl_context` argument is not allowed for the http scheme')
        self._ssl_context = ssl_context
        self._proxy_ssl_context = proxy_ssl_context
        self._proxy_headers = enforce_headers(proxy_headers, name='proxy_headers')
        if proxy_auth is not None:
            username = enforce_bytes(proxy_auth[0], name='proxy_auth')
            password = enforce_bytes(proxy_auth[1], name='proxy_auth')
            authorization = build_auth_header(username, password)
            self._proxy_headers = [(b'Proxy-Authorization', authorization)] + self._proxy_headers

    def create_connection(self, origin: Origin) -> AsyncConnectionInterface:
        if origin.scheme == b'http':
            return AsyncForwardHTTPConnection(proxy_origin=self._proxy_url.origin, proxy_headers=self._proxy_headers, remote_origin=origin, keepalive_expiry=self._keepalive_expiry, network_backend=self._network_backend, proxy_ssl_context=self._proxy_ssl_context)
        return AsyncTunnelHTTPConnection(proxy_origin=self._proxy_url.origin, proxy_headers=self._proxy_headers, remote_origin=origin, ssl_context=self._ssl_context, proxy_ssl_context=self._proxy_ssl_context, keepalive_expiry=self._keepalive_expiry, http1=self._http1, http2=self._http2, network_backend=self._network_backend)