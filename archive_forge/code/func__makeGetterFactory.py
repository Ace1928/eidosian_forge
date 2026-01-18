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
def _makeGetterFactory(url, factoryFactory, contextFactory=None, *args, **kwargs):
    """
    Create and connect an HTTP page getting factory.

    Any additional positional or keyword arguments are used when calling
    C{factoryFactory}.

    @param factoryFactory: Factory factory that is called with C{url}, C{args}
        and C{kwargs} to produce the getter

    @param contextFactory: Context factory to use when creating a secure
        connection, defaulting to L{None}

    @return: The factory created by C{factoryFactory}
    """
    uri = URI.fromBytes(_ensureValidURI(url.strip()))
    factory = factoryFactory(url, *args, **kwargs)
    from twisted.internet import reactor
    if uri.scheme == b'https':
        from twisted.internet import ssl
        if contextFactory is None:
            contextFactory = ssl.ClientContextFactory()
        reactor.connectSSL(nativeString(uri.host), uri.port, factory, contextFactory)
    else:
        reactor.connectTCP(nativeString(uri.host), uri.port, factory)
    return factory