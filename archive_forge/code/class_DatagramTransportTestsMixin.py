import socket
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import defer, error
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import DatagramProtocol
from twisted.internet.test.connectionmixins import LogObserverMixin, findFreePort
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python import context
from twisted.python.log import ILogContext, err
from twisted.test.test_udp import GoodClient, Server
from twisted.trial.unittest import SkipTest
class DatagramTransportTestsMixin(LogObserverMixin):
    """
    Mixin defining tests which apply to any port/datagram based transport.
    """

    def test_startedListeningLogMessage(self):
        """
        When a port starts, a message including a description of the associated
        protocol is logged.
        """
        loggedMessages = self.observe()
        reactor = self.buildReactor()

        @implementer(ILoggingContext)
        class SomeProtocol(DatagramProtocol):

            def logPrefix(self):
                return 'Crazy Protocol'
        protocol = SomeProtocol()
        p = self.getListeningPort(reactor, protocol)
        expectedMessage = 'Crazy Protocol starting on %d' % (p.getHost().port,)
        self.assertEqual((expectedMessage,), loggedMessages[0]['message'])

    def test_connectionLostLogMessage(self):
        """
        When a connection is lost a message is logged containing an
        address identifying the port and the fact that it was closed.
        """
        loggedMessages = self.observe()
        reactor = self.buildReactor()
        p = self.getListeningPort(reactor, DatagramProtocol())
        expectedMessage = f'(UDP Port {p.getHost().port} Closed)'

        def stopReactor(ignored):
            reactor.stop()

        def doStopListening():
            del loggedMessages[:]
            maybeDeferred(p.stopListening).addCallback(stopReactor)
        reactor.callWhenRunning(doStopListening)
        self.runReactor(reactor)
        self.assertEqual((expectedMessage,), loggedMessages[0]['message'])

    def test_stopProtocolScheduling(self):
        """
        L{DatagramProtocol.stopProtocol} is called asynchronously (ie, not
        re-entrantly) when C{stopListening} is used to stop the datagram
        transport.
        """

        class DisconnectingProtocol(DatagramProtocol):
            started = False
            stopped = False
            inStartProtocol = False
            stoppedInStart = False

            def startProtocol(self):
                self.started = True
                self.inStartProtocol = True
                self.transport.stopListening()
                self.inStartProtocol = False

            def stopProtocol(self):
                self.stopped = True
                self.stoppedInStart = self.inStartProtocol
                reactor.stop()
        reactor = self.buildReactor()
        protocol = DisconnectingProtocol()
        self.getListeningPort(reactor, protocol)
        self.runReactor(reactor)
        self.assertTrue(protocol.started)
        self.assertTrue(protocol.stopped)
        self.assertFalse(protocol.stoppedInStart)