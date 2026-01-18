from __future__ import annotations
import collections
import os
import warnings
import zlib
from dataclasses import dataclass
from functools import wraps
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlunparse as _urlunparse
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, protocol, task
from twisted.internet.abstract import isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import HostnameEndpoint, wrapClientTLS
from twisted.internet.interfaces import IOpenSSLContextFactory, IProtocol
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import (
from twisted.python.failure import Failure
from twisted.web import error, http
from twisted.web._newclient import _ensureValidMethod, _ensureValidURI
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web._newclient import (
from twisted.web.error import SchemeNotSupported
@implementer(IAgent)
class ProxyAgent(_AgentBase):
    """
    An HTTP agent able to cross HTTP proxies.

    @ivar _proxyEndpoint: The endpoint used to connect to the proxy.

    @since: 11.1
    """

    def __init__(self, endpoint, reactor=None, pool=None):
        if reactor is None:
            from twisted.internet import reactor
        _AgentBase.__init__(self, reactor, pool)
        self._proxyEndpoint = endpoint

    def request(self, method, uri, headers=None, bodyProducer=None):
        """
        Issue a new request via the configured proxy.
        """
        uri = _ensureValidURI(uri.strip())
        key = ('http-proxy', self._proxyEndpoint)
        return self._requestWithEndpoint(key, self._proxyEndpoint, method, URI.fromBytes(uri), headers, bodyProducer, uri)