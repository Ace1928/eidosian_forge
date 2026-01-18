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
@implementer(IPolicyForHTTPS)
class HostnameCachingHTTPSPolicy:
    """
    IPolicyForHTTPS that wraps a L{IPolicyForHTTPS} and caches the created
    L{IOpenSSLClientConnectionCreator}.

    This policy will cache up to C{cacheSize}
    L{client connection creators <twisted.internet.interfaces.
    IOpenSSLClientConnectionCreator>} for reuse in subsequent requests to
    the same hostname.

    @ivar _policyForHTTPS: See C{policyforHTTPS} parameter of L{__init__}.

    @ivar _cache: A cache associating hostnames to their
        L{client connection creators <twisted.internet.interfaces.
        IOpenSSLClientConnectionCreator>}.
    @type _cache: L{collections.OrderedDict}

    @ivar _cacheSize: See C{cacheSize} parameter of L{__init__}.

    @since: Twisted 19.2.0
    """

    def __init__(self, policyforHTTPS, cacheSize=20):
        """
        @param policyforHTTPS: The IPolicyForHTTPS to wrap.
        @type policyforHTTPS: L{IPolicyForHTTPS}

        @param cacheSize: The maximum size of the hostname cache.
        @type cacheSize: L{int}
        """
        self._policyForHTTPS = policyforHTTPS
        self._cache = collections.OrderedDict()
        self._cacheSize = cacheSize

    def creatorForNetloc(self, hostname, port):
        """
        Create a L{client connection creator
        <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>} for a
        given network location and cache it for future use.

        @param hostname: The hostname part of the URI.
        @type hostname: L{bytes}

        @param port: The port part of the URI.
        @type port: L{int}

        @return: a connection creator with appropriate verification
            restrictions set
        @rtype: L{client connection creator
            <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>}
        """
        host = hostname.decode('ascii')
        try:
            creator = self._cache.pop(host)
        except KeyError:
            creator = self._policyForHTTPS.creatorForNetloc(hostname, port)
        self._cache[host] = creator
        if len(self._cache) > self._cacheSize:
            self._cache.popitem(last=False)
        return creator