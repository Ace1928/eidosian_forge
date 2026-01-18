import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class TimeoutFactoryTests(unittest.TestCase):
    """
    Tests for L{policies.TimeoutFactory}.
    """

    def setUp(self):
        """
        Create a testable, deterministic clock, and a set of
        server factory/protocol/transport.
        """
        self.clock = task.Clock()
        wrappedFactory = protocol.ServerFactory()
        wrappedFactory.protocol = SimpleProtocol
        self.factory = TestableTimeoutFactory(self.clock, wrappedFactory, 3)
        self.proto = self.factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 12345))
        self.transport = StringTransportWithDisconnection()
        self.transport.protocol = self.proto
        self.proto.makeConnection(self.transport)
        self.wrappedProto = self.proto.wrappedProtocol

    def test_timeout(self):
        """
        Make sure that when a TimeoutFactory accepts a connection, it will
        time out that connection if no data is read or written within the
        timeout period.
        """
        self.clock.pump([0.0, 0.5, 1.0, 1.0, 0.4])
        self.assertFalse(self.wrappedProto.disconnected)
        self.clock.pump([0.0, 0.2])
        self.assertTrue(self.wrappedProto.disconnected)

    def test_sendAvoidsTimeout(self):
        """
        Make sure that writing data to a transport from a protocol
        constructed by a TimeoutFactory resets the timeout countdown.
        """
        self.clock.pump([0.0, 0.5, 1.0])
        self.assertFalse(self.wrappedProto.disconnected)
        self.proto.write(b'bytes bytes bytes')
        self.clock.pump([0.0, 1.0, 1.0])
        self.assertFalse(self.wrappedProto.disconnected)
        self.proto.writeSequence([b'bytes'] * 3)
        self.clock.pump([0.0, 1.0, 1.0])
        self.assertFalse(self.wrappedProto.disconnected)
        self.clock.pump([0.0, 2.0])
        self.assertTrue(self.wrappedProto.disconnected)

    def test_receiveAvoidsTimeout(self):
        """
        Make sure that receiving data also resets the timeout countdown.
        """
        self.clock.pump([0.0, 1.0, 0.5])
        self.assertFalse(self.wrappedProto.disconnected)
        self.proto.dataReceived(b'bytes bytes bytes')
        self.clock.pump([0.0, 1.0, 1.0])
        self.assertFalse(self.wrappedProto.disconnected)
        self.clock.pump([0.0, 1.0, 1.0])
        self.assertTrue(self.wrappedProto.disconnected)