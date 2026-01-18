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
class DatagramProtocolTests(unittest.TestCase):
    """
    Test various aspects of L{dns.DNSDatagramProtocol}.
    """

    def setUp(self):
        """
        Create a L{dns.DNSDatagramProtocol} with a deterministic clock.
        """
        self.clock = task.Clock()
        self.controller = TestController()
        self.proto = dns.DNSDatagramProtocol(self.controller)
        transport = proto_helpers.FakeDatagramTransport()
        self.proto.makeConnection(transport)
        self.proto.callLater = self.clock.callLater

    def test_truncatedPacket(self):
        """
        Test that when a short datagram is received, datagramReceived does
        not raise an exception while processing it.
        """
        self.proto.datagramReceived(b'', address.IPv4Address('UDP', '127.0.0.1', 12345))
        self.assertEqual(self.controller.messages, [])

    def test_malformedMessage(self):
        """
        Test that when an unparsable message is received, datagramReceived does
        not raise an exception while processing it.
        """
        unparsable = b'\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x02\x11WWWWWWWWWW-XXXXXX\x08_arduino\x04_tcp\x05local\x00\x00\xff\x80\x01\xc07\x00\x0c\x00\x01\x00\x00\x11\x94\x00\x02\xc0V\xc0V\x00!\x00\x01\x00\x00\x11\x94\x00\x08\x00\x00\x00\x00 J\xc0\x8f\xc0V\x00\x10\x00\x01\x00\x00\x11\x94\x00K\x0eauth_upload=no board="ESP8266_WEMOS_D1MINILITE"\rssh_upload=no\x0ctcp_check=no\xc0\x8f\x00\x01\x00\x01\x00\x00\x00x\x00\x04\xc0\xa8\x01)'
        self.proto.datagramReceived(unparsable, address.IPv4Address('UDP', '127.0.0.1', 12345))
        self.assertEqual(self.controller.messages, [])

    def test_simpleQuery(self):
        """
        Test content received after a query.
        """
        d = self.proto.query(('127.0.0.1', 21345), [dns.Query(b'foo')])
        self.assertEqual(len(self.proto.liveMessages.keys()), 1)
        m = dns.Message()
        m.id = next(iter(self.proto.liveMessages.keys()))
        m.answers = [dns.RRHeader(payload=dns.Record_A(address='1.2.3.4'))]

        def cb(result):
            self.assertEqual(result.answers[0].payload.dottedQuad(), '1.2.3.4')
        d.addCallback(cb)
        self.proto.datagramReceived(m.toStr(), ('127.0.0.1', 21345))
        return d

    def test_queryTimeout(self):
        """
        Test that query timeouts after some seconds.
        """
        d = self.proto.query(('127.0.0.1', 21345), [dns.Query(b'foo')])
        self.assertEqual(len(self.proto.liveMessages), 1)
        self.clock.advance(10)
        self.assertFailure(d, dns.DNSQueryTimeoutError)
        self.assertEqual(len(self.proto.liveMessages), 0)
        return d

    def test_writeError(self):
        """
        Exceptions raised by the transport's write method should be turned into
        C{Failure}s passed to errbacks of the C{Deferred} returned by
        L{DNSDatagramProtocol.query}.
        """

        def writeError(message, addr):
            raise RuntimeError('bar')
        self.proto.transport.write = writeError
        d = self.proto.query(('127.0.0.1', 21345), [dns.Query(b'foo')])
        return self.assertFailure(d, RuntimeError)

    def test_listenError(self):
        """
        Exception L{CannotListenError} raised by C{listenUDP} should be turned
        into a C{Failure} passed to errback of the C{Deferred} returned by
        L{DNSDatagramProtocol.query}.
        """

        def startListeningError():
            raise CannotListenError(None, None, None)
        self.proto.startListening = startListeningError
        self.proto.transport = None
        d = self.proto.query(('127.0.0.1', 21345), [dns.Query(b'foo')])
        return self.assertFailure(d, CannotListenError)

    def test_receiveMessageNotInLiveMessages(self):
        """
        When receiving a message whose id is not in
        L{DNSDatagramProtocol.liveMessages} or L{DNSDatagramProtocol.resends},
        the message will be received by L{DNSDatagramProtocol.controller}.
        """
        message = dns.Message()
        message.id = 1
        message.answers = [dns.RRHeader(payload=dns.Record_A(address='1.2.3.4'))]
        self.proto.datagramReceived(message.toStr(), ('127.0.0.1', 21345))
        self.assertEqual(self.controller.messages[-1][0].toStr(), message.toStr())