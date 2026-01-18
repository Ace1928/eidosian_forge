import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
@skipIf(not interfaces.IReactorUDP(reactor, None), 'This reactor does not support UDP')
class ReactorShutdownInteractionTests(TestCase):
    """Test reactor shutdown interaction"""
    if not interfaces.IReactorUDP(reactor, None):
        skip = 'This reactor does not support UDP'

    def setUp(self):
        """Start a UDP port"""
        self.server = Server()
        self.port = reactor.listenUDP(0, self.server, interface='127.0.0.1')

    def tearDown(self):
        """Stop the UDP port"""
        return self.port.stopListening()

    def testShutdownFromDatagramReceived(self):
        """Test reactor shutdown while in a recvfrom() loop"""
        finished = defer.Deferred()
        pr = self.server.packetReceived = defer.Deferred()

        def pktRece(ignored):
            self.server.transport.connectionLost()
            reactor.callLater(0, finished.callback, None)
        pr.addCallback(pktRece)

        def flushErrors(ignored):
            self.flushLoggedErrors()
        finished.addCallback(flushErrors)
        self.server.transport.write(b'\x00' * 64, ('127.0.0.1', self.server.transport.getHost().port))
        return finished