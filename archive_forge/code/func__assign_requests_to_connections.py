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