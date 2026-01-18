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
class ConnectionLostTests(TestCase, ContextGeneratingMixin):
    """
    SSL connection closing tests.
    """
    if interfaces.IReactorSSL(reactor, None) is None:
        skip = 'Reactor does not support SSL, cannot run SSL tests'

    def testImmediateDisconnect(self):
        org = 'twisted.test.test_ssl'
        self.setupServerAndClient((org, org + ', client'), {}, (org, org + ', server'), {})
        serverProtocolFactory = protocol.ServerFactory()
        serverProtocolFactory.protocol = protocol.Protocol
        self.serverPort = serverPort = reactor.listenSSL(0, serverProtocolFactory, self.serverCtxFactory)
        clientProtocolFactory = protocol.ClientFactory()
        clientProtocolFactory.protocol = ImmediatelyDisconnectingProtocol
        clientProtocolFactory.connectionDisconnected = defer.Deferred()
        reactor.connectSSL('127.0.0.1', serverPort.getHost().port, clientProtocolFactory, self.clientCtxFactory)
        return clientProtocolFactory.connectionDisconnected.addCallback(lambda ignoredResult: self.serverPort.stopListening())

    def test_bothSidesLoseConnection(self):
        """
        Both sides of SSL connection close connection; the connections should
        close cleanly, and only after the underlying TCP connection has
        disconnected.
        """

        @implementer(interfaces.IHandshakeListener)
        class CloseAfterHandshake(protocol.Protocol):
            gotData = False

            def __init__(self):
                self.done = defer.Deferred()

            def handshakeCompleted(self):
                self.transport.loseConnection()

            def connectionLost(self, reason):
                self.done.errback(reason)
                del self.done
        org = 'twisted.test.test_ssl'
        self.setupServerAndClient((org, org + ', client'), {}, (org, org + ', server'), {})
        serverProtocol = CloseAfterHandshake()
        serverProtocolFactory = protocol.ServerFactory()
        serverProtocolFactory.protocol = lambda: serverProtocol
        serverPort = reactor.listenSSL(0, serverProtocolFactory, self.serverCtxFactory)
        self.addCleanup(serverPort.stopListening)
        clientProtocol = CloseAfterHandshake()
        clientProtocolFactory = protocol.ClientFactory()
        clientProtocolFactory.protocol = lambda: clientProtocol
        reactor.connectSSL('127.0.0.1', serverPort.getHost().port, clientProtocolFactory, self.clientCtxFactory)

        def checkResult(failure):
            failure.trap(ConnectionDone)
        return defer.gatherResults([clientProtocol.done.addErrback(checkResult), serverProtocol.done.addErrback(checkResult)])

    def testFailedVerify(self):
        org = 'twisted.test.test_ssl'
        self.setupServerAndClient((org, org + ', client'), {}, (org, org + ', server'), {})

        def verify(*a):
            return False
        self.clientCtxFactory.getContext().set_verify(SSL.VERIFY_PEER, verify)
        serverConnLost = defer.Deferred()
        serverProtocol = protocol.Protocol()
        serverProtocol.connectionLost = serverConnLost.callback
        serverProtocolFactory = protocol.ServerFactory()
        serverProtocolFactory.protocol = lambda: serverProtocol
        self.serverPort = serverPort = reactor.listenSSL(0, serverProtocolFactory, self.serverCtxFactory)
        clientConnLost = defer.Deferred()
        clientProtocol = protocol.Protocol()
        clientProtocol.connectionLost = clientConnLost.callback
        clientProtocolFactory = protocol.ClientFactory()
        clientProtocolFactory.protocol = lambda: clientProtocol
        reactor.connectSSL('127.0.0.1', serverPort.getHost().port, clientProtocolFactory, self.clientCtxFactory)
        dl = defer.DeferredList([serverConnLost, clientConnLost], consumeErrors=True)
        return dl.addCallback(self._cbLostConns)

    def _cbLostConns(self, results):
        (sSuccess, sResult), (cSuccess, cResult) = results
        self.assertFalse(sSuccess)
        self.assertFalse(cSuccess)
        acceptableErrors = [SSL.Error]
        if platform.isWindows():
            from twisted.internet.error import ConnectionLost
            acceptableErrors.append(ConnectionLost)
        sResult.trap(*acceptableErrors)
        cResult.trap(*acceptableErrors)
        return self.serverPort.stopListening()