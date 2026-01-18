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
class H2Agent:

    def __init__(self, reactor: ReactorBase, pool: H2ConnectionPool, context_factory: BrowserLikePolicyForHTTPS=BrowserLikePolicyForHTTPS(), connect_timeout: Optional[float]=None, bind_address: Optional[bytes]=None) -> None:
        self._reactor = reactor
        self._pool = pool
        self._context_factory = AcceptableProtocolsContextFactory(context_factory, acceptable_protocols=[b'h2'])
        self.endpoint_factory = _StandardEndpointFactory(self._reactor, self._context_factory, connect_timeout, bind_address)

    def get_endpoint(self, uri: URI):
        return self.endpoint_factory.endpointForURI(uri)

    def get_key(self, uri: URI) -> Tuple:
        """
        Arguments:
            uri - URI obtained directly from request URL
        """
        return (uri.scheme, uri.host, uri.port)

    def request(self, request: Request, spider: Spider) -> Deferred:
        uri = URI.fromBytes(bytes(request.url, encoding='utf-8'))
        try:
            endpoint = self.get_endpoint(uri)
        except SchemeNotSupported:
            return defer.fail(Failure())
        key = self.get_key(uri)
        d = self._pool.get_connection(key, uri, endpoint)
        d.addCallback(lambda conn: conn.request(request, spider))
        return d