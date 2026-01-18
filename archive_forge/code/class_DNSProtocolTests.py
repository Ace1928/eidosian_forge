import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class DNSProtocolTests(unittest.TestCase):
    """
    Test various aspects of L{dns.DNSProtocol}.
    """

    def setUp(self):
        """
        Create a L{dns.DNSProtocol} with a deterministic clock.
        """
        self.clock = task.Clock()
        self.controller = TestTCPController()
        self.proto = dns.DNSProtocol(self.controller)
        self.proto.makeConnection(proto_helpers.StringTransport())
        self.proto.callLater = self.clock.callLater

    def test_connectionTracking(self):
        """
        L{dns.DNSProtocol} calls its controller's C{connectionMade}
        method with itself when it is connected to a transport and its
        controller's C{connectionLost} method when it is disconnected.
        """
        self.assertEqual(self.controller.connections, [self.proto])
        self.proto.connectionLost(Failure(ConnectionDone('Fake Connection Done')))
        self.assertEqual(self.controller.connections, [])

    def test_queryTimeout(self):
        """
        Test that query timeouts after some seconds.
        """
        d = self.proto.query([dns.Query(b'foo')])
        self.assertEqual(len(self.proto.liveMessages), 1)
        self.clock.advance(60)
        self.assertFailure(d, dns.DNSQueryTimeoutError)
        self.assertEqual(len(self.proto.liveMessages), 0)
        return d

    def test_simpleQuery(self):
        """
        Test content received after a query.
        """
        d = self.proto.query([dns.Query(b'foo')])
        self.assertEqual(len(self.proto.liveMessages.keys()), 1)
        m = dns.Message()
        m.id = next(iter(self.proto.liveMessages.keys()))
        m.answers = [dns.RRHeader(payload=dns.Record_A(address='1.2.3.4'))]

        def cb(result):
            self.assertEqual(result.answers[0].payload.dottedQuad(), '1.2.3.4')
        d.addCallback(cb)
        s = m.toStr()
        s = struct.pack('!H', len(s)) + s
        self.proto.dataReceived(s)
        return d

    def test_writeError(self):
        """
        Exceptions raised by the transport's write method should be turned into
        C{Failure}s passed to errbacks of the C{Deferred} returned by
        L{DNSProtocol.query}.
        """

        def writeError(message):
            raise RuntimeError('bar')
        self.proto.transport.write = writeError
        d = self.proto.query([dns.Query(b'foo')])
        return self.assertFailure(d, RuntimeError)

    def test_receiveMessageNotInLiveMessages(self):
        """
        When receiving a message whose id is not in L{DNSProtocol.liveMessages}
        the message will be received by L{DNSProtocol.controller}.
        """
        message = dns.Message()
        message.id = 1
        message.answers = [dns.RRHeader(payload=dns.Record_A(address='1.2.3.4'))]
        string = message.toStr()
        string = struct.pack('!H', len(string)) + string
        self.proto.dataReceived(string)
        self.assertEqual(self.controller.messages[-1][0].toStr(), message.toStr())