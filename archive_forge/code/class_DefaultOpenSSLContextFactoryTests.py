import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
class DefaultOpenSSLContextFactoryTests(TestCase):
    """
    Tests for L{ssl.DefaultOpenSSLContextFactory}.
    """
    if interfaces.IReactorSSL(reactor, None) is None:
        skip = 'Reactor does not support SSL, cannot run SSL tests'

    def setUp(self):
        self.contextFactory = ssl.DefaultOpenSSLContextFactory(certPath, certPath, _contextFactory=FakeContext)
        self.context = self.contextFactory.getContext()

    def test_method(self):
        """
        L{ssl.DefaultOpenSSLContextFactory.getContext} returns an SSL context
        which can use SSLv3 or TLSv1 but not SSLv2.
        """
        self.assertEqual(self.context._method, SSL.TLS_METHOD)
        self.assertEqual(self.context._options & SSL.OP_NO_SSLv2, SSL.OP_NO_SSLv2)
        self.assertFalse(self.context._options & SSL.OP_NO_TLSv1_2)

    def test_missingCertificateFile(self):
        """
        Instantiating L{ssl.DefaultOpenSSLContextFactory} with a certificate
        filename which does not identify an existing file results in the
        initializer raising L{OpenSSL.SSL.Error}.
        """
        self.assertRaises(SSL.Error, ssl.DefaultOpenSSLContextFactory, certPath, self.mktemp())

    def test_missingPrivateKeyFile(self):
        """
        Instantiating L{ssl.DefaultOpenSSLContextFactory} with a private key
        filename which does not identify an existing file results in the
        initializer raising L{OpenSSL.SSL.Error}.
        """
        self.assertRaises(SSL.Error, ssl.DefaultOpenSSLContextFactory, self.mktemp(), certPath)