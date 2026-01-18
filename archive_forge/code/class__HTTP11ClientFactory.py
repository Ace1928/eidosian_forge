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
class _HTTP11ClientFactory(protocol.Factory):
    """
    A factory for L{HTTP11ClientProtocol}, used by L{HTTPConnectionPool}.

    @ivar _quiescentCallback: The quiescent callback to be passed to protocol
        instances, used to return them to the connection pool.

    @ivar _metadata: Metadata about the low-level connection details,
        used to make the repr more useful.

    @since: 11.1
    """

    def __init__(self, quiescentCallback, metadata):
        self._quiescentCallback = quiescentCallback
        self._metadata = metadata

    def __repr__(self) -> str:
        return '_HTTP11ClientFactory({}, {})'.format(self._quiescentCallback, self._metadata)

    def buildProtocol(self, addr):
        return HTTP11ClientProtocol(self._quiescentCallback)