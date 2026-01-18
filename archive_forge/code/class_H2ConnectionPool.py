from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from twisted.internet import defer
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import HostnameEndpoint
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from scrapy.core.downloader.contextfactory import AcceptableProtocolsContextFactory
from scrapy.core.http2.protocol import H2ClientFactory, H2ClientProtocol
from scrapy.http.request import Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
class H2ConnectionPool:

    def __init__(self, reactor: ReactorBase, settings: Settings) -> None:
        self._reactor = reactor
        self.settings = settings
        self._connections: Dict[Tuple, H2ClientProtocol] = {}
        self._pending_requests: Dict[Tuple, Deque[Deferred]] = {}

    def get_connection(self, key: Tuple, uri: URI, endpoint: HostnameEndpoint) -> Deferred:
        if key in self._pending_requests:
            d: Deferred = Deferred()
            self._pending_requests[key].append(d)
            return d
        conn = self._connections.get(key, None)
        if conn:
            return defer.succeed(conn)
        return self._new_connection(key, uri, endpoint)

    def _new_connection(self, key: Tuple, uri: URI, endpoint: HostnameEndpoint) -> Deferred:
        self._pending_requests[key] = deque()
        conn_lost_deferred: Deferred = Deferred()
        conn_lost_deferred.addCallback(self._remove_connection, key)
        factory = H2ClientFactory(uri, self.settings, conn_lost_deferred)
        conn_d = endpoint.connect(factory)
        conn_d.addCallback(self.put_connection, key)
        d: Deferred = Deferred()
        self._pending_requests[key].append(d)
        return d

    def put_connection(self, conn: H2ClientProtocol, key: Tuple) -> H2ClientProtocol:
        self._connections[key] = conn
        pending_requests = self._pending_requests.pop(key, None)
        while pending_requests:
            d = pending_requests.popleft()
            d.callback(conn)
        return conn

    def _remove_connection(self, errors: List[BaseException], key: Tuple) -> None:
        self._connections.pop(key)
        pending_requests = self._pending_requests.pop(key, None)
        while pending_requests:
            d = pending_requests.popleft()
            d.errback(ResponseFailed(errors))

    def close_connections(self) -> None:
        """Close all the HTTP/2 connections and remove them from pool

        Returns:
            Deferred that fires when all connections have been closed
        """
        for conn in self._connections.values():
            assert conn.transport is not None
            conn.transport.abortConnection()