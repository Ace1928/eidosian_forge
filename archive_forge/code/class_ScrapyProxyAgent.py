import ipaddress
import logging
import re
from contextlib import suppress
from io import BytesIO
from time import time
from urllib.parse import urldefrag, urlunparse
from twisted.internet import defer, protocol, ssl
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.internet.error import TimeoutError
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.http import PotentialDataLoss, _DataLoss
from twisted.web.http_headers import Headers as TxHeaders
from twisted.web.iweb import UNKNOWN_LENGTH, IBodyProducer
from zope.interface import implementer
from scrapy import signals
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.exceptions import StopDownload
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.python import to_bytes, to_unicode
class ScrapyProxyAgent(Agent):

    def __init__(self, reactor, proxyURI, connectTimeout=None, bindAddress=None, pool=None):
        super().__init__(reactor=reactor, connectTimeout=connectTimeout, bindAddress=bindAddress, pool=pool)
        self._proxyURI = URI.fromBytes(proxyURI)

    def request(self, method, uri, headers=None, bodyProducer=None):
        """
        Issue a new request via the configured proxy.
        """
        return self._requestWithEndpoint(key=('http-proxy', self._proxyURI.host, self._proxyURI.port), endpoint=self._getEndpoint(self._proxyURI), method=method, parsedURI=URI.fromBytes(uri), headers=headers, bodyProducer=bodyProducer, requestPath=uri)