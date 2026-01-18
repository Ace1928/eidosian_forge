import re
from time import time
from typing import Optional, Tuple
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from twisted.internet import defer
from twisted.internet.protocol import ClientFactory
from twisted.web.http import HTTPClient
from scrapy import Request
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes, to_unicode
class ScrapyHTTPPageGetter(HTTPClient):
    delimiter = b'\n'

    def connectionMade(self):
        self.headers = Headers()
        self.sendCommand(self.factory.method, self.factory.path)
        for key, values in self.factory.headers.items():
            for value in values:
                self.sendHeader(key, value)
        self.endHeaders()
        if self.factory.body is not None:
            self.transport.write(self.factory.body)

    def lineReceived(self, line):
        return HTTPClient.lineReceived(self, line.rstrip())

    def handleHeader(self, key, value):
        self.headers.appendlist(key, value)

    def handleStatus(self, version, status, message):
        self.factory.gotStatus(version, status, message)

    def handleEndHeaders(self):
        self.factory.gotHeaders(self.headers)

    def connectionLost(self, reason):
        self._connection_lost_reason = reason
        HTTPClient.connectionLost(self, reason)
        self.factory.noPage(reason)

    def handleResponse(self, response):
        if self.factory.method.upper() == b'HEAD':
            self.factory.page(b'')
        elif self.length is not None and self.length > 0:
            self.factory.noPage(self._connection_lost_reason)
        else:
            self.factory.page(response)
        self.transport.loseConnection()

    def timeout(self):
        self.transport.loseConnection()
        if self.factory.url.startswith(b'https'):
            self.transport.stopProducing()
        self.factory.noPage(defer.TimeoutError(f'Getting {self.factory.url} took longer than {self.factory.timeout} seconds.'))