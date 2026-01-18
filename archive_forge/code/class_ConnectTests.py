import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class ConnectTests(unittest.TestCase):
    """
    Tests for SOCKS and SOCKSv4a connect requests using the L{SOCKSv4} protocol.
    """

    def setUp(self):
        self.sock = SOCKSv4Driver()
        self.sock.transport = StringTCPTransport()
        self.sock.connectionMade()
        self.sock.reactor = FakeResolverReactor({b'localhost': '127.0.0.1'})

    def tearDown(self):
        outgoing = self.sock.driver_outgoing
        if outgoing is not None:
            self.assertTrue(outgoing.transport.stringTCPTransport_closing, 'Outgoing SOCKS connections need to be closed.')

    def test_simple(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 1, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 34) + socket.inet_aton('1.2.3.4'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNotNone(self.sock.driver_outgoing)
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(self.sock.driver_outgoing.transport.value(), b'hello, world')
        self.sock.driver_outgoing.dataReceived(b'hi there')
        self.assertEqual(self.sock.transport.value(), b'hi there')
        self.sock.connectionLost('fake reason')

    def test_socks4aSuccessfulResolution(self):
        """
        If the destination IP address has zeros for the first three octets and
        non-zero for the fourth octet, the client is attempting a v4a
        connection.  A hostname is specified after the user ID string and the
        server connects to the address that hostname resolves to.

        @see: U{http://en.wikipedia.org/wiki/SOCKS#SOCKS_4a_protocol}
        """
        clientRequest = struct.pack('!BBH', 4, 1, 34) + socket.inet_aton('0.0.0.1') + b'fooBAZ\x00' + b'localhost\x00'
        for byte in iterbytes(clientRequest):
            self.sock.dataReceived(byte)
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 34) + socket.inet_aton('127.0.0.1'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNotNone(self.sock.driver_outgoing)
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(self.sock.driver_outgoing.transport.value(), b'hello, world')
        self.sock.driver_outgoing.dataReceived(b'hi there')
        self.assertEqual(self.sock.transport.value(), b'hi there')
        self.sock.connectionLost('fake reason')

    def test_socks4aFailedResolution(self):
        """
        Failed hostname resolution on a SOCKSv4a packet results in a 91 error
        response and the connection getting closed.
        """
        clientRequest = struct.pack('!BBH', 4, 1, 34) + socket.inet_aton('0.0.0.1') + b'fooBAZ\x00' + b'failinghost\x00'
        for byte in iterbytes(clientRequest):
            self.sock.dataReceived(byte)
        sent = self.sock.transport.value()
        self.assertEqual(sent, struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
        self.assertTrue(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNone(self.sock.driver_outgoing)

    def test_accessDenied(self):
        self.sock.authorize = lambda code, server, port, user: 0
        self.sock.dataReceived(struct.pack('!BBH', 4, 1, 4242) + socket.inet_aton('10.2.3.4') + b'fooBAR' + b'\x00')
        self.assertEqual(self.sock.transport.value(), struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
        self.assertTrue(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNone(self.sock.driver_outgoing)

    def test_eofRemote(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 1, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        self.sock.transport.clear()
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(self.sock.driver_outgoing.transport.value(), b'hello, world')
        self.sock.driver_outgoing.transport.loseConnection()
        self.sock.driver_outgoing.connectionLost('fake reason')

    def test_eofLocal(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 1, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        self.sock.transport.clear()
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(self.sock.driver_outgoing.transport.value(), b'hello, world')
        self.sock.connectionLost('fake reason')