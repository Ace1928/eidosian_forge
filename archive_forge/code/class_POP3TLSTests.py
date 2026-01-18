import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
@skipIf(not ClientTLSContext, 'OpenSSL not present')
@skipIf(not interfaces.IReactorSSL(reactor, None), 'OpenSSL not present')
class POP3TLSTests(TestCase):
    """
    Tests for POP3Client's support for TLS connections.
    """

    def test_startTLS(self):
        """
        POP3Client.startTLS starts a TLS session over its existing TCP
        connection.
        """
        sf = TLSServerFactory()
        sf.protocol.output = [[b'+OK'], [b'+OK', b'STLS', b'.'], [b'+OK'], [b'+OK', b'.'], [b'+OK']]
        sf.protocol.context = ServerTLSContext()
        port = reactor.listenTCP(0, sf, interface='127.0.0.1')
        self.addCleanup(port.stopListening)
        H = port.getHost().host
        P = port.getHost().port
        connLostDeferred = defer.Deferred()
        cp = SimpleClient(defer.Deferred(), ClientTLSContext())

        def connectionLost(reason):
            SimpleClient.connectionLost(cp, reason)
            connLostDeferred.callback(None)
        cp.connectionLost = connectionLost
        cf = protocol.ClientFactory()
        cf.protocol = lambda: cp
        conn = reactor.connectTCP(H, P, cf)

        def cbConnected(ignored):
            log.msg('Connected to server; starting TLS')
            return cp.startTLS()

        def cbStartedTLS(ignored):
            log.msg('Started TLS; disconnecting')
            return cp.quit()

        def cbDisconnected(ign):
            log.msg('Disconnected; asserting correct input received')
            self.assertEqual(sf.input, [b'CAPA', b'STLS', b'CAPA', b'QUIT'])

        def cleanup(result):
            log.msg('Asserted correct input; disconnecting client and shutting down server')
            conn.disconnect()
            return connLostDeferred
        cp.deferred.addCallback(cbConnected)
        cp.deferred.addCallback(cbStartedTLS)
        cp.deferred.addCallback(cbDisconnected)
        cp.deferred.addBoth(cleanup)
        return cp.deferred