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
class ClientContextFactoryTests(TestCase):
    """
    Tests for L{ssl.ClientContextFactory}.
    """
    if interfaces.IReactorSSL(reactor, None) is None:
        skip = 'Reactor does not support SSL, cannot run SSL tests'

    def setUp(self):
        self.contextFactory = ssl.ClientContextFactory()
        self.contextFactory._contextFactory = FakeContext
        self.context = self.contextFactory.getContext()

    def test_method(self):
        """
        L{ssl.ClientContextFactory.getContext} returns a context which can use
        TLSv1.2 or 1.3 but nothing earlier.
        """
        self.assertEqual(self.context._method, SSL.TLS_METHOD)
        self.assertEqual(self.context._options & SSL.OP_NO_SSLv2, SSL.OP_NO_SSLv2)
        self.assertTrue(self.context._options & SSL.OP_NO_SSLv3)
        self.assertTrue(self.context._options & SSL.OP_NO_TLSv1)