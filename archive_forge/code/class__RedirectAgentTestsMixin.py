from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
class _RedirectAgentTestsMixin(testMixinClass):
    """
    Test cases mixin for L{RedirectAgentTests} and
    L{BrowserLikeRedirectAgentTests}.
    """
    agent: IAgent
    reactor: MemoryReactorClock
    protocol: StubHTTPProtocol

    def test_noRedirect(self):
        """
        L{client.RedirectAgent} behaves like L{client.Agent} if the response
        doesn't contain a redirect.
        """
        deferred = self.agent.request(b'GET', b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers()
        response = Response((b'HTTP', 1, 1), 200, b'OK', headers, None)
        res.callback(response)
        self.assertEqual(0, len(self.protocol.requests))
        result = self.successResultOf(deferred)
        self.assertIdentical(response, result)
        self.assertIdentical(result.previousResponse, None)

    def _testRedirectDefault(self, code: int, crossScheme: bool=False, crossDomain: bool=False, crossPort: bool=False, requestHeaders: Optional[Headers]=None) -> Request:
        """
        When getting a redirect, L{client.RedirectAgent} follows the URL
        specified in the L{Location} header field and make a new request.

        @param code: HTTP status code.
        """
        startDomain = b'example.com'
        startScheme = b'https' if ssl is not None else b'http'
        startPort = 80 if startScheme == b'http' else 443
        self.agent.request(b'GET', startScheme + b'://' + startDomain + b'/foo', headers=requestHeaders)
        host, port = self.reactor.tcpClients.pop()[:2]
        self.assertEqual(EXAMPLE_COM_IP, host)
        self.assertEqual(startPort, port)
        req, res = self.protocol.requests.pop()
        targetScheme = startScheme
        targetDomain = startDomain
        targetPort = startPort
        if crossScheme:
            if ssl is None:
                raise SkipTest("Cross-scheme redirects can't be tested without TLS support.")
            targetScheme = b'https' if startScheme == b'http' else b'http'
            targetPort = 443 if startPort == 80 else 80
        portSyntax = b''
        if crossPort:
            targetPort = 8443
            portSyntax = b':8443'
        targetDomain = b'example.net' if crossDomain else startDomain
        locationValue = targetScheme + b'://' + targetDomain + portSyntax + b'/bar'
        headers = http_headers.Headers({b'location': [locationValue]})
        response = Response((b'HTTP', 1, 1), code, b'OK', headers, None)
        res.callback(response)
        req2, res2 = self.protocol.requests.pop()
        self.assertEqual(b'GET', req2.method)
        self.assertEqual(b'/bar', req2.uri)
        host, port = self.reactor.tcpClients.pop()[:2]
        self.assertEqual(EXAMPLE_NET_IP if crossDomain else EXAMPLE_COM_IP, host)
        self.assertEqual(targetPort, port)
        return req2

    def test_redirect301(self):
        """
        L{client.RedirectAgent} follows redirects on status code 301.
        """
        self._testRedirectDefault(301)

    def test_redirect301Scheme(self):
        """
        L{client.RedirectAgent} follows cross-scheme redirects.
        """
        self._testRedirectDefault(301, crossScheme=True)

    def test_redirect302(self):
        """
        L{client.RedirectAgent} follows redirects on status code 302.
        """
        self._testRedirectDefault(302)

    def test_redirect307(self):
        """
        L{client.RedirectAgent} follows redirects on status code 307.
        """
        self._testRedirectDefault(307)

    def test_redirect308(self):
        """
        L{client.RedirectAgent} follows redirects on status code 308.
        """
        self._testRedirectDefault(308)

    def _sensitiveHeadersTest(self, expectedHostHeader: bytes=b'example.com', **crossKwargs: bool) -> None:
        """
        L{client.RedirectAgent} scrubs sensitive headers when redirecting
        between differing origins.
        """
        sensitiveHeaderValues = {b'authorization': [b'sensitive-authnz'], b'cookie': [b'sensitive-cookie-data'], b'cookie2': [b'sensitive-cookie2-data'], b'proxy-authorization': [b'sensitive-proxy-auth'], b'wWw-auThentiCate': [b'sensitive-authn'], b'x-custom-sensitive': [b'sensitive-custom']}
        otherHeaderValues = {b'x-random-header': [b'x-random-value']}
        allHeaders = Headers({**sensitiveHeaderValues, **otherHeaderValues})
        redirected = self._testRedirectDefault(301, requestHeaders=allHeaders)

        def normHeaders(headers: Headers) -> Dict[bytes, Sequence[bytes]]:
            return {k.lower(): v for k, v in headers.getAllRawHeaders()}
        sameOriginHeaders = normHeaders(redirected.headers)
        self.assertEquals(sameOriginHeaders, {b'host': [b'example.com'], **normHeaders(allHeaders)})
        redirectedElsewhere = self._testRedirectDefault(301, **crossKwargs, requestHeaders=Headers({**sensitiveHeaderValues, **otherHeaderValues}))
        otherOriginHeaders = normHeaders(redirectedElsewhere.headers)
        self.assertEquals(otherOriginHeaders, {b'host': [expectedHostHeader], **normHeaders(Headers(otherHeaderValues))})

    def test_crossDomainHeaders(self) -> None:
        """
        L{client.RedirectAgent} scrubs sensitive headers when redirecting
        between differing domains.
        """
        self._sensitiveHeadersTest(crossDomain=True, expectedHostHeader=b'example.net')

    def test_crossPortHeaders(self) -> None:
        """
        L{client.RedirectAgent} scrubs sensitive headers when redirecting
        between differing ports.
        """
        self._sensitiveHeadersTest(crossPort=True, expectedHostHeader=b'example.com:8443')

    def test_crossSchemeHeaders(self) -> None:
        """
        L{client.RedirectAgent} scrubs sensitive headers when redirecting
        between differing schemes.
        """
        self._sensitiveHeadersTest(crossScheme=True)

    def _testRedirectToGet(self, code, method):
        """
        L{client.RedirectAgent} changes the method to I{GET} when getting
        a redirect on a non-I{GET} request.

        @param code: HTTP status code.

        @param method: HTTP request method.
        """
        self.agent.request(method, b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers({b'location': [b'http://example.com/bar']})
        response = Response((b'HTTP', 1, 1), code, b'OK', headers, None)
        res.callback(response)
        req2, res2 = self.protocol.requests.pop()
        self.assertEqual(b'GET', req2.method)
        self.assertEqual(b'/bar', req2.uri)

    def test_redirect303(self):
        """
        L{client.RedirectAgent} changes the method to I{GET} when getting a 303
        redirect on a I{POST} request.
        """
        self._testRedirectToGet(303, b'POST')

    def test_noLocationField(self):
        """
        If no L{Location} header field is found when getting a redirect,
        L{client.RedirectAgent} fails with a L{ResponseFailed} error wrapping a
        L{error.RedirectWithNoLocation} exception.
        """
        deferred = self.agent.request(b'GET', b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers()
        response = Response((b'HTTP', 1, 1), 301, b'OK', headers, None)
        res.callback(response)
        fail = self.failureResultOf(deferred, client.ResponseFailed)
        fail.value.reasons[0].trap(error.RedirectWithNoLocation)
        self.assertEqual(b'http://example.com/foo', fail.value.reasons[0].value.uri)
        self.assertEqual(301, fail.value.response.code)

    def _testPageRedirectFailure(self, code, method):
        """
        When getting a redirect on an unsupported request method,
        L{client.RedirectAgent} fails with a L{ResponseFailed} error wrapping
        a L{error.PageRedirect} exception.

        @param code: HTTP status code.

        @param method: HTTP request method.
        """
        deferred = self.agent.request(method, b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers()
        response = Response((b'HTTP', 1, 1), code, b'OK', headers, None)
        res.callback(response)
        fail = self.failureResultOf(deferred, client.ResponseFailed)
        fail.value.reasons[0].trap(error.PageRedirect)
        self.assertEqual(b'http://example.com/foo', fail.value.reasons[0].value.location)
        self.assertEqual(code, fail.value.response.code)

    def test_307OnPost(self):
        """
        When getting a 307 redirect on a I{POST} request,
        L{client.RedirectAgent} fails with a L{ResponseFailed} error wrapping
        a L{error.PageRedirect} exception.
        """
        self._testPageRedirectFailure(307, b'POST')

    def test_redirectLimit(self):
        """
        If the limit of redirects specified to L{client.RedirectAgent} is
        reached, the deferred fires with L{ResponseFailed} error wrapping
        a L{InfiniteRedirection} exception.
        """
        agent = self.buildAgentForWrapperTest(self.reactor)
        redirectAgent = client.RedirectAgent(agent, 1)
        deferred = redirectAgent.request(b'GET', b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers({b'location': [b'http://example.com/bar']})
        response = Response((b'HTTP', 1, 1), 302, b'OK', headers, None)
        res.callback(response)
        req2, res2 = self.protocol.requests.pop()
        response2 = Response((b'HTTP', 1, 1), 302, b'OK', headers, None)
        res2.callback(response2)
        fail = self.failureResultOf(deferred, client.ResponseFailed)
        fail.value.reasons[0].trap(error.InfiniteRedirection)
        self.assertEqual(b'http://example.com/foo', fail.value.reasons[0].value.location)
        self.assertEqual(302, fail.value.response.code)

    def _testRedirectURI(self, uri, location, finalURI):
        """
        When L{client.RedirectAgent} encounters a relative redirect I{URI}, it
        is resolved against the request I{URI} before following the redirect.

        @param uri: Request URI.

        @param location: I{Location} header redirect URI.

        @param finalURI: Expected final URI.
        """
        self.agent.request(b'GET', uri)
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers({b'location': [location]})
        response = Response((b'HTTP', 1, 1), 302, b'OK', headers, None)
        res.callback(response)
        req2, res2 = self.protocol.requests.pop()
        self.assertEqual(b'GET', req2.method)
        self.assertEqual(finalURI, req2.absoluteURI)

    def test_relativeURI(self):
        """
        L{client.RedirectAgent} resolves and follows relative I{URI}s in
        redirects, preserving query strings.
        """
        self._testRedirectURI(b'http://example.com/foo/bar', b'baz', b'http://example.com/foo/baz')
        self._testRedirectURI(b'http://example.com/foo/bar', b'/baz', b'http://example.com/baz')
        self._testRedirectURI(b'http://example.com/foo/bar', b'/baz?a', b'http://example.com/baz?a')

    def test_relativeURIPreserveFragments(self):
        """
        L{client.RedirectAgent} resolves and follows relative I{URI}s in
        redirects, preserving fragments in way that complies with the HTTP 1.1
        bis draft.

        @see: U{https://tools.ietf.org/html/draft-ietf-httpbis-p2-semantics-22#section-7.1.2}
        """
        self._testRedirectURI(b'http://example.com/foo/bar#frag', b'/baz?a', b'http://example.com/baz?a#frag')
        self._testRedirectURI(b'http://example.com/foo/bar', b'/baz?a#frag2', b'http://example.com/baz?a#frag2')

    def test_relativeURISchemeRelative(self):
        """
        L{client.RedirectAgent} resolves and follows scheme relative I{URI}s in
        redirects, replacing the hostname and port when required.
        """
        self._testRedirectURI(b'http://example.com/foo/bar', b'//foo.com/baz', b'http://foo.com/baz')
        self._testRedirectURI(b'http://example.com/foo/bar', b'//foo.com:81/baz', b'http://foo.com:81/baz')

    def test_responseHistory(self):
        """
        L{Response.response} references the previous L{Response} from
        a redirect, or L{None} if there was no previous response.
        """
        agent = self.buildAgentForWrapperTest(self.reactor)
        redirectAgent = client.RedirectAgent(agent)
        deferred = redirectAgent.request(b'GET', b'http://example.com/foo')
        redirectReq, redirectRes = self.protocol.requests.pop()
        headers = http_headers.Headers({b'location': [b'http://example.com/bar']})
        redirectResponse = Response((b'HTTP', 1, 1), 302, b'OK', headers, None)
        redirectRes.callback(redirectResponse)
        req, res = self.protocol.requests.pop()
        response = Response((b'HTTP', 1, 1), 200, b'OK', headers, None)
        res.callback(response)
        finalResponse = self.successResultOf(deferred)
        self.assertIdentical(finalResponse.previousResponse, redirectResponse)
        self.assertIdentical(redirectResponse.previousResponse, None)