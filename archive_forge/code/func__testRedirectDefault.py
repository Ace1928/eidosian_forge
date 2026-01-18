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