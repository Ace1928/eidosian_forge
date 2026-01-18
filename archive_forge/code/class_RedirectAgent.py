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
class RedirectAgent:
    """
    An L{Agent} wrapper which handles HTTP redirects.

    The implementation is rather strict: 301 and 302 behaves like 307, not
    redirecting automatically on methods different from I{GET} and I{HEAD}.

    See L{BrowserLikeRedirectAgent} for a redirecting Agent that behaves more
    like a web browser.

    @param redirectLimit: The maximum number of times the agent is allowed to
        follow redirects before failing with a L{error.InfiniteRedirection}.

    @param sensitiveHeaderNames: An iterable of C{bytes} enumerating the names
        of headers that must not be transmitted when redirecting to a different
        origins.  These will be consulted in addition to the protocol-specified
        set of headers that contain sensitive information.

    @cvar _redirectResponses: A L{list} of HTTP status codes to be redirected
        for I{GET} and I{HEAD} methods.

    @cvar _seeOtherResponses: A L{list} of HTTP status codes to be redirected
        for any method and the method altered to I{GET}.

    @since: 11.1
    """
    _redirectResponses = [http.MOVED_PERMANENTLY, http.FOUND, http.TEMPORARY_REDIRECT, http.PERMANENT_REDIRECT]
    _seeOtherResponses = [http.SEE_OTHER]

    def __init__(self, agent: IAgent, redirectLimit: int=20, sensitiveHeaderNames: Iterable[bytes]=()):
        self._agent = agent
        self._redirectLimit = redirectLimit
        sensitive = {_canonicalHeaderName(each) for each in sensitiveHeaderNames}
        sensitive.update(_defaultSensitiveHeaders)
        self._sensitiveHeaderNames = sensitive

    def request(self, method, uri, headers=None, bodyProducer=None):
        """
        Send a client request following HTTP redirects.

        @see: L{Agent.request}.
        """
        deferred = self._agent.request(method, uri, headers, bodyProducer)
        return deferred.addCallback(self._handleResponse, method, uri, headers, 0)

    def _resolveLocation(self, requestURI, location):
        """
        Resolve the redirect location against the request I{URI}.

        @type requestURI: C{bytes}
        @param requestURI: The request I{URI}.

        @type location: C{bytes}
        @param location: The redirect location.

        @rtype: C{bytes}
        @return: Final resolved I{URI}.
        """
        return _urljoin(requestURI, location)

    def _handleRedirect(self, response, method, uri, headers, redirectCount):
        """
        Handle a redirect response, checking the number of redirects already
        followed, and extracting the location header fields.
        """
        if redirectCount >= self._redirectLimit:
            err = error.InfiniteRedirection(response.code, b'Infinite redirection detected', location=uri)
            raise ResponseFailed([Failure(err)], response)
        locationHeaders = response.headers.getRawHeaders(b'location', [])
        if not locationHeaders:
            err = error.RedirectWithNoLocation(response.code, b'No location header field', uri)
            raise ResponseFailed([Failure(err)], response)
        location = self._resolveLocation(uri, locationHeaders[0])
        if headers:
            parsedURI = URI.fromBytes(uri)
            parsedLocation = URI.fromBytes(location)
            sameOrigin = parsedURI.scheme == parsedLocation.scheme and parsedURI.host == parsedLocation.host and (parsedURI.port == parsedLocation.port)
            if not sameOrigin:
                headers = Headers({rawName: rawValue for rawName, rawValue in headers.getAllRawHeaders() if rawName not in self._sensitiveHeaderNames})
        deferred = self._agent.request(method, location, headers)

        def _chainResponse(newResponse):
            newResponse.setPreviousResponse(response)
            return newResponse
        deferred.addCallback(_chainResponse)
        return deferred.addCallback(self._handleResponse, method, uri, headers, redirectCount + 1)

    def _handleResponse(self, response, method, uri, headers, redirectCount):
        """
        Handle the response, making another request if it indicates a redirect.
        """
        if response.code in self._redirectResponses:
            if method not in (b'GET', b'HEAD'):
                err = error.PageRedirect(response.code, location=uri)
                raise ResponseFailed([Failure(err)], response)
            return self._handleRedirect(response, method, uri, headers, redirectCount)
        elif response.code in self._seeOtherResponses:
            return self._handleRedirect(response, b'GET', uri, headers, redirectCount)
        return response