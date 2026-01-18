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
class WebClientContextFactoryTests(TestCase):
    """
    Tests for the context factory wrapper for web clients
    L{twisted.web.client.WebClientContextFactory}.
    """

    def setUp(self):
        """
        Get WebClientContextFactory while quashing its deprecation warning.
        """
        from twisted.web.client import WebClientContextFactory
        self.warned = self.flushWarnings([WebClientContextFactoryTests.setUp])
        self.webClientContextFactory = WebClientContextFactory

    def test_deprecated(self):
        """
        L{twisted.web.client.WebClientContextFactory} is deprecated.  Importing
        it displays a warning.
        """
        self.assertEqual(len(self.warned), 1)
        [warning] = self.warned
        self.assertEqual(warning['category'], DeprecationWarning)
        self.assertEqual(warning['message'], getDeprecationWarningString(self.webClientContextFactory, Version('Twisted', 14, 0, 0), replacement=BrowserLikePolicyForHTTPS).replace(';', ':'))

    @skipIf(sslPresent, 'SSL Present.')
    def test_missingSSL(self):
        """
        If C{getContext} is called and SSL is not available, raise
        L{NotImplementedError}.
        """
        self.assertRaises(NotImplementedError, self.webClientContextFactory().getContext, b'example.com', 443)

    @skipIf(not sslPresent, 'SSL not present, cannot run SSL tests.')
    def test_returnsContext(self):
        """
        If SSL is present, C{getContext} returns a L{OpenSSL.SSL.Context}.
        """
        ctx = self.webClientContextFactory().getContext('example.com', 443)
        self.assertIsInstance(ctx, ssl.SSL.Context)

    @skipIf(not sslPresent, 'SSL not present, cannot run SSL tests.')
    def test_setsTrustRootOnContextToDefaultTrustRoot(self):
        """
        The L{CertificateOptions} has C{trustRoot} set to the default trust
        roots.
        """
        ctx = self.webClientContextFactory()
        certificateOptions = ctx._getCertificateOptions('example.com', 443)
        self.assertIsInstance(certificateOptions.trustRoot, ssl.OpenSSLDefaultPaths)