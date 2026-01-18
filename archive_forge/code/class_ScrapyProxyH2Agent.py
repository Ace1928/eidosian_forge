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
class ScrapyProxyH2Agent(H2Agent):

    def __init__(self, reactor: ReactorBase, proxy_uri: URI, pool: H2ConnectionPool, context_factory: BrowserLikePolicyForHTTPS=BrowserLikePolicyForHTTPS(), connect_timeout: Optional[float]=None, bind_address: Optional[bytes]=None) -> None:
        super().__init__(reactor=reactor, pool=pool, context_factory=context_factory, connect_timeout=connect_timeout, bind_address=bind_address)
        self._proxy_uri = proxy_uri

    def get_endpoint(self, uri: URI):
        return self.endpoint_factory.endpointForURI(self._proxy_uri)

    def get_key(self, uri: URI) -> Tuple:
        """We use the proxy uri instead of uri obtained from request url"""
        return ('http-proxy', self._proxy_uri.host, self._proxy_uri.port)