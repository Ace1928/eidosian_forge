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
class _FakeStdlibRequest(_RequestBase):
    """
    A fake L{urllib.request.Request} object for L{cookiejar} to work with.

    @see: U{urllib.request.Request
        <https://docs.python.org/3/library/urllib.request.html#urllib.request.Request>}

    @ivar uri: Request URI.

    @ivar headers: Request headers.

    @ivar type: The scheme of the URI.

    @ivar host: The host[:port] of the URI.

    @since: 11.1
    """
    uri: str
    type: str
    host: str
    _twistedHeaders: Headers

    def __init__(self, uri: bytes) -> None:
        """
        Create a fake  request.

        @param uri: Request URI.
        """
        self.uri = nativeString(uri)
        self._twistedHeaders = Headers()
        _uri = URI.fromBytes(uri)
        self.type = nativeString(_uri.scheme)
        self.host = nativeString(_uri.host)
        if (_uri.scheme, _uri.port) not in ((b'http', 80), (b'https', 443)):
            self.host += ':' + str(_uri.port)
        self.origin_req_host = nativeString(_uri.host)
        self.unverifiable = False

    def has_header(self, header):
        return self._twistedHeaders.hasHeader(networkString(header))

    def add_unredirected_header(self, name, value):
        self._twistedHeaders.addRawHeader(networkString(name), networkString(value))

    def get_full_url(self):
        return self.uri

    def get_header(self, name, default=None):
        headers = self._twistedHeaders.getRawHeaders(networkString(name), default)
        if headers is not None:
            headers = [nativeString(x) for x in headers]
            return headers[0]
        return None

    def get_host(self):
        return self.host

    def get_type(self):
        return self.type

    def is_unverifiable(self):
        return False