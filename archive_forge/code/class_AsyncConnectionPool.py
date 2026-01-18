import ssl
import sys
from types import TracebackType
from typing import AsyncIterable, AsyncIterator, Iterable, List, Optional, Type
from .._backends.auto import AutoBackend
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ConnectionNotAvailable, UnsupportedProtocol
from .._models import Origin, Request, Response
from .._synchronization import AsyncEvent, AsyncShieldCancellation, AsyncThreadLock
from .connection import AsyncHTTPConnection
from .interfaces import AsyncConnectionInterface, AsyncRequestInterface
class AsyncConnectionPool(AsyncRequestInterface):
    """
    A connection pool for making HTTP requests.
    """

    def __init__(self, ssl_context: Optional[ssl.SSLContext]=None, max_connections: Optional[int]=10, max_keepalive_connections: Optional[int]=None, keepalive_expiry: Optional[float]=None, http1: bool=True, http2: bool=False, retries: int=0, local_address: Optional[str]=None, uds: Optional[str]=None, network_backend: Optional[AsyncNetworkBackend]=None, socket_options: Optional[Iterable[SOCKET_OPTION]]=None) -> None:
        """
        A connection pool for making HTTP requests.

        Parameters:
            ssl_context: An SSL context to use for verifying connections.
                If not specified, the default `httpcore.default_ssl_context()`
                will be used.
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
            retries: The maximum number of retries when trying to establish a
                connection.
            local_address: Local address to connect from. Can also be used to connect
                using a particular address family. Using `local_address="0.0.0.0"`
                will connect using an `AF_INET` address (IPv4), while using
                `local_address="::"` will connect using an `AF_INET6` address (IPv6).
            uds: Path to a Unix Domain Socket to use instead of TCP sockets.
            network_backend: A backend instance to use for handling network I/O.
            socket_options: Socket options that have to be included
             in the TCP socket when the connection was established.
        """
        self._ssl_context = ssl_context
        self._max_connections = sys.maxsize if max_connections is None else max_connections
        self._max_keepalive_connections = sys.maxsize if max_keepalive_connections is None else max_keepalive_connections
        self._max_keepalive_connections = min(self._max_connections, self._max_keepalive_connections)
        self._keepalive_expiry = keepalive_expiry
        self._http1 = http1
        self._http2 = http2
        self._retries = retries
        self._local_address = local_address
        self._uds = uds
        self._network_backend = AutoBackend() if network_backend is None else network_backend
        self._socket_options = socket_options
        self._connections: List[AsyncConnectionInterface] = []
        self._requests: List[AsyncPoolRequest] = []
        self._optional_thread_lock = AsyncThreadLock()

    def create_connection(self, origin: Origin) -> AsyncConnectionInterface:
        return AsyncHTTPConnection(origin=origin, ssl_context=self._ssl_context, keepalive_expiry=self._keepalive_expiry, http1=self._http1, http2=self._http2, retries=self._retries, local_address=self._local_address, uds=self._uds, network_backend=self._network_backend, socket_options=self._socket_options)

    @property
    def connections(self) -> List[AsyncConnectionInterface]:
        """
        Return a list of the connections currently in the pool.

        For example:

        ```python
        >>> pool.connections
        [
            <AsyncHTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 6]>,
            <AsyncHTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 9]> ,
            <AsyncHTTPConnection ['http://example.com:80', HTTP/1.1, IDLE, Request Count: 1]>,
        ]
        ```
        """
        return list(self._connections)

    async def handle_async_request(self, request: Request) -> Response:
        """
        Send an HTTP request, and return an HTTP response.

        This is the core implementation that is called into by `.request()` or `.stream()`.
        """
        scheme = request.url.scheme.decode()
        if scheme == '':
            raise UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol.")
        if scheme not in ('http', 'https', 'ws', 'wss'):
            raise UnsupportedProtocol(f"Request URL has an unsupported protocol '{scheme}://'.")
        timeouts = request.extensions.get('timeout', {})
        timeout = timeouts.get('pool', None)
        with self._optional_thread_lock:
            pool_request = AsyncPoolRequest(request)
            self._requests.append(pool_request)
        try:
            while True:
                with self._optional_thread_lock:
                    closing = self._assign_requests_to_connections()
                await self._close_connections(closing)
                connection = await pool_request.wait_for_connection(timeout=timeout)
                try:
                    response = await connection.handle_async_request(pool_request.request)
                except ConnectionNotAvailable:
                    pool_request.clear_connection()
                else:
                    break
        except BaseException as exc:
            with self._optional_thread_lock:
                self._requests.remove(pool_request)
                closing = self._assign_requests_to_connections()
            await self._close_connections(closing)
            raise exc from None
        assert isinstance(response.stream, AsyncIterable)
        return Response(status=response.status, headers=response.headers, content=PoolByteStream(stream=response.stream, pool_request=pool_request, pool=self), extensions=response.extensions)

    def _assign_requests_to_connections(self) -> List[AsyncConnectionInterface]:
        """
        Manage the state of the connection pool, assigning incoming
        requests to connections as available.

        Called whenever a new request is added or removed from the pool.

        Any closing connections are returned, allowing the I/O for closing
        those connections to be handled seperately.
        """
        closing_connections = []
        for connection in list(self._connections):
            if connection.is_closed():
                self._connections.remove(connection)
            elif connection.has_expired():
                self._connections.remove(connection)
                closing_connections.append(connection)
            elif connection.is_idle() and len([connection.is_idle() for connection in self._connections]) > self._max_keepalive_connections:
                self._connections.remove(connection)
                closing_connections.append(connection)
        queued_requests = [request for request in self._requests if request.is_queued()]
        for pool_request in queued_requests:
            origin = pool_request.request.url.origin
            avilable_connections = [connection for connection in self._connections if connection.can_handle_request(origin) and connection.is_available()]
            idle_connections = [connection for connection in self._connections if connection.is_idle()]
            if avilable_connections:
                connection = avilable_connections[0]
                pool_request.assign_to_connection(connection)
            elif len(self._connections) < self._max_connections:
                connection = self.create_connection(origin)
                self._connections.append(connection)
                pool_request.assign_to_connection(connection)
            elif idle_connections:
                connection = idle_connections[0]
                self._connections.remove(connection)
                closing_connections.append(connection)
                connection = self.create_connection(origin)
                self._connections.append(connection)
                pool_request.assign_to_connection(connection)
        return closing_connections

    async def _close_connections(self, closing: List[AsyncConnectionInterface]) -> None:
        with AsyncShieldCancellation():
            for connection in closing:
                await connection.aclose()

    async def aclose(self) -> None:
        with self._optional_thread_lock:
            closing_connections = list(self._connections)
            self._connections = []
        await self._close_connections(closing_connections)

    async def __aenter__(self) -> 'AsyncConnectionPool':
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        with self._optional_thread_lock:
            request_is_queued = [request.is_queued() for request in self._requests]
            connection_is_idle = [connection.is_idle() for connection in self._connections]
            num_active_requests = request_is_queued.count(False)
            num_queued_requests = request_is_queued.count(True)
            num_active_connections = connection_is_idle.count(False)
            num_idle_connections = connection_is_idle.count(True)
        requests_info = f'Requests: {num_active_requests} active, {num_queued_requests} queued'
        connection_info = f'Connections: {num_active_connections} active, {num_idle_connections} idle'
        return f'<{class_name} [{requests_info} | {connection_info}]>'