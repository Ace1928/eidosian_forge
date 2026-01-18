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
class _DeprecatedToCurrentPolicyForHTTPS:
    """
    Adapt a web context factory to a normal context factory.

    @ivar _webContextFactory: An object providing a getContext method with
        C{hostname} and C{port} arguments.
    @type _webContextFactory: L{WebClientContextFactory} (or object with a
        similar C{getContext} method).
    """

    def __init__(self, webContextFactory):
        """
        Wrap a web context factory in an L{IPolicyForHTTPS}.

        @param webContextFactory: An object providing a getContext method with
            C{hostname} and C{port} arguments.
        @type webContextFactory: L{WebClientContextFactory} (or object with a
            similar C{getContext} method).
        """
        self._webContextFactory = webContextFactory

    def creatorForNetloc(self, hostname, port):
        """
        Called the wrapped web context factory's C{getContext} method with a
        hostname and port number and return the resulting context object.

        @param hostname: The hostname part of the URI.
        @type hostname: L{bytes}

        @param port: The port part of the URI.
        @type port: L{int}

        @return: A context factory.
        @rtype: L{IOpenSSLContextFactory}
        """
        context = self._webContextFactory.getContext(hostname, port)
        return _ContextFactoryWithContext(context)