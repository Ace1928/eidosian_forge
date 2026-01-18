import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class OpenSSLOptionsTestsMixin:
    """
    A mixin for L{OpenSSLOptions} test cases creates client and server
    certificates, signs them with a CA, and provides a L{loopback}
    that creates TLS a connections with them.
    """
    if skipSSL:
        skip = skipSSL
    serverPort = clientConn = None
    onServerLost = onClientLost = None

    def setUp(self):
        """
        Create class variables of client and server certificates.
        """
        self.sKey, self.sCert = makeCertificate(O=b'Server Test Certificate', CN=b'server')
        self.cKey, self.cCert = makeCertificate(O=b'Client Test Certificate', CN=b'client')
        self.caCert1 = makeCertificate(O=b'CA Test Certificate 1', CN=b'ca1')[1]
        self.caCert2 = makeCertificate(O=b'CA Test Certificate', CN=b'ca2')[1]
        self.caCerts = [self.caCert1, self.caCert2]
        self.extraCertChain = self.caCerts

    def tearDown(self):
        if self.serverPort is not None:
            self.serverPort.stopListening()
        if self.clientConn is not None:
            self.clientConn.disconnect()
        L = []
        if self.onServerLost is not None:
            L.append(self.onServerLost)
        if self.onClientLost is not None:
            L.append(self.onClientLost)
        return defer.DeferredList(L, consumeErrors=True)

    def loopback(self, serverCertOpts, clientCertOpts, onServerLost=None, onClientLost=None, onData=None):
        if onServerLost is None:
            self.onServerLost = onServerLost = defer.Deferred()
        if onClientLost is None:
            self.onClientLost = onClientLost = defer.Deferred()
        if onData is None:
            onData = defer.Deferred()
        serverFactory = protocol.ServerFactory()
        serverFactory.protocol = DataCallbackProtocol
        serverFactory.onLost = onServerLost
        serverFactory.onData = onData
        clientFactory = protocol.ClientFactory()
        clientFactory.protocol = WritingProtocol
        clientFactory.onLost = onClientLost
        self.serverPort = reactor.listenSSL(0, serverFactory, serverCertOpts)
        self.clientConn = reactor.connectSSL('127.0.0.1', self.serverPort.getHost().port, clientFactory, clientCertOpts)