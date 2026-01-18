import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class LimitTotalConnectionsFactoryTests(unittest.TestCase):
    """Tests for policies.LimitTotalConnectionsFactory"""

    def testConnectionCounting(self):
        factory = policies.LimitTotalConnectionsFactory()
        factory.protocol = protocol.Protocol
        self.assertEqual(0, factory.connectionCount)
        p1 = factory.buildProtocol(None)
        self.assertEqual(1, factory.connectionCount)
        p2 = factory.buildProtocol(None)
        self.assertEqual(2, factory.connectionCount)
        p1.connectionLost(None)
        self.assertEqual(1, factory.connectionCount)
        p2.connectionLost(None)
        self.assertEqual(0, factory.connectionCount)

    def testConnectionLimiting(self):
        factory = policies.LimitTotalConnectionsFactory()
        factory.protocol = protocol.Protocol
        factory.connectionLimit = 1
        p = factory.buildProtocol(None)
        self.assertIsNotNone(p)
        self.assertEqual(1, factory.connectionCount)
        self.assertIsNone(factory.buildProtocol(None))
        self.assertEqual(1, factory.connectionCount)

        class OverflowProtocol(protocol.Protocol):

            def connectionMade(self):
                factory.overflowed = True
        factory.overflowProtocol = OverflowProtocol
        factory.overflowed = False
        op = factory.buildProtocol(None)
        op.makeConnection(None)
        self.assertTrue(factory.overflowed)
        self.assertEqual(2, factory.connectionCount)
        p.connectionLost(None)
        self.assertEqual(1, factory.connectionCount)
        op.connectionLost(None)
        self.assertEqual(0, factory.connectionCount)