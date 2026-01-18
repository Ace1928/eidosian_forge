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
class _AgentBase:
    """
    Base class offering common facilities for L{Agent}-type classes.

    @ivar _reactor: The C{IReactorTime} implementation which will be used by
        the pool, and perhaps by subclasses as well.

    @ivar _pool: The L{HTTPConnectionPool} used to manage HTTP connections.
    """

    def __init__(self, reactor, pool):
        if pool is None:
            pool = HTTPConnectionPool(reactor, False)
        self._reactor = reactor
        self._pool = pool

    def _computeHostValue(self, scheme, host, port):
        """
        Compute the string to use for the value of the I{Host} header, based on
        the given scheme, host name, and port number.
        """
        if isIPv6Address(nativeString(host)):
            host = b'[' + host + b']'
        if (scheme, port) in ((b'http', 80), (b'https', 443)):
            return host
        return b'%b:%d' % (host, port)

    def _requestWithEndpoint(self, key, endpoint, method, parsedURI, headers, bodyProducer, requestPath):
        """
        Issue a new request, given the endpoint and the path sent as part of
        the request.
        """
        if not isinstance(method, bytes):
            raise TypeError(f'method={method!r} is {type(method)}, but must be bytes')
        method = _ensureValidMethod(method)
        if headers is None:
            headers = Headers()
        if not headers.hasHeader(b'host'):
            headers = headers.copy()
            headers.addRawHeader(b'host', self._computeHostValue(parsedURI.scheme, parsedURI.host, parsedURI.port))
        d = self._pool.getConnection(key, endpoint)

        def cbConnected(proto):
            return proto.request(Request._construct(method, requestPath, headers, bodyProducer, persistent=self._pool.persistent, parsedURI=parsedURI))
        d.addCallback(cbConnected)
        return d