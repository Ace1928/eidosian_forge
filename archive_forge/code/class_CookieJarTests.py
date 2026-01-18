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
class CookieJarTests(TestCase, CookieTestsMixin):
    """
    Tests for L{twisted.web.client._FakeStdlibResponse} and
    L{twisted.web.client._FakeStdlibRequest}'s interactions with L{CookieJar}
    instances.
    """

    def makeCookieJar(self) -> tuple[CookieJar, tuple[client._FakeStdlibRequest, client._FakeStdlibResponse]]:
        """
        @return: a L{CookieJar} with some sample cookies
        """
        cookieJar = CookieJar()
        reqres = self.addCookies(cookieJar, b'http://example.com:1234/foo?bar', [b'foo=1; cow=moo; Path=/foo; Comment=hello', b'bar=2; Comment=goodbye'])
        return (cookieJar, reqres)

    def test_extractCookies(self) -> None:
        """
        L{CookieJar.extract_cookies} extracts cookie information from our
        stdlib-compatibility wrappers, L{client._FakeStdlibRequest} and
        L{client._FakeStdlibResponse}.
        """
        jar = self.makeCookieJar()[0]
        cookies = {c.name: c for c in jar}
        cookie = cookies['foo']
        self.assertEqual(cookie.version, 0)
        self.assertEqual(cookie.name, 'foo')
        self.assertEqual(cookie.value, '1')
        self.assertEqual(cookie.path, '/foo')
        self.assertEqual(cookie.comment, 'hello')
        self.assertEqual(cookie.get_nonstandard_attr('cow'), 'moo')
        cookie = cookies['bar']
        self.assertEqual(cookie.version, 0)
        self.assertEqual(cookie.name, 'bar')
        self.assertEqual(cookie.value, '2')
        self.assertEqual(cookie.path, '/')
        self.assertEqual(cookie.comment, 'goodbye')
        self.assertIdentical(cookie.get_nonstandard_attr('cow'), None)

    def test_sendCookie(self) -> None:
        """
        L{CookieJar.add_cookie_header} adds a cookie header to a Twisted
        request via our L{client._FakeStdlibRequest} wrapper.
        """
        jar, (request, response) = self.makeCookieJar()
        self.assertIdentical(request.get_header('Cookie', None), None)
        jar.add_cookie_header(request)
        self.assertEqual(list(request._twistedHeaders.getAllRawHeaders()), [(b'Cookie', [b'foo=1; bar=2'])])