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
class BrowserLikeRedirectAgent(RedirectAgent):
    """
    An L{Agent} wrapper which handles HTTP redirects in the same fashion as web
    browsers.

    Unlike L{RedirectAgent}, the implementation is more relaxed: 301 and 302
    behave like 303, redirecting automatically on any method and altering the
    redirect request to a I{GET}.

    @see: L{RedirectAgent}

    @since: 13.1
    """
    _redirectResponses = [http.TEMPORARY_REDIRECT]
    _seeOtherResponses = [http.MOVED_PERMANENTLY, http.FOUND, http.SEE_OTHER, http.PERMANENT_REDIRECT]