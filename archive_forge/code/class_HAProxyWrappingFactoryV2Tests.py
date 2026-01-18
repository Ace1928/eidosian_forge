from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
class HAProxyWrappingFactoryV2Tests(unittest.TestCase):
    """
    Test L{twisted.protocols.haproxy.HAProxyWrappingFactory} with v2 PROXY
    headers.
    """
    IPV4HEADER = b'\r\n\r\n\x00\r\nQUIT\n!\x11\x00\x0c\x7f\x00\x00\x01\x7f\x00\x00\x01\x1f\x90"\xb8'
    IPV6HEADER = b'\r\n\r\n\x00\r\nQUIT\n!!\x00$\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x1f\x90"\xb8'
    _SOCK_PATH = b'/home/tests/mysockets/sock' + b'\x00' * 82
    UNIXHEADER = b'\r\n\r\n\x00\r\nQUIT\n!1\x00\xd8' + _SOCK_PATH + _SOCK_PATH

    def test_invalidHeaderDisconnects(self) -> None:
        """
        Test if invalid headers result in connectionLost events.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        transport.protocol = proto
        proto.makeConnection(transport)
        proto.dataReceived(b'\x00' + self.IPV4HEADER[1:])
        self.assertFalse(transport.connected)

    def test_validIPv4HeaderResolves_getPeerHost(self) -> None:
        """
        Test if IPv4 headers result in the correct host and peer data.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(self.IPV4HEADER)
        self.assertEqual(proto.getPeer().host, '127.0.0.1')
        self.assertEqual(proto.getPeer().port, 8080)
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().host, '127.0.0.1')
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().port, 8080)
        self.assertEqual(proto.getHost().host, '127.0.0.1')
        self.assertEqual(proto.getHost().port, 8888)
        self.assertEqual(proto.wrappedProtocol.transport.getHost().host, '127.0.0.1')
        self.assertEqual(proto.wrappedProtocol.transport.getHost().port, 8888)

    def test_validIPv6HeaderResolves_getPeerHost(self) -> None:
        """
        Test if IPv6 headers result in the correct host and peer data.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv4Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(self.IPV6HEADER)
        self.assertEqual(proto.getPeer().host, '0:0:0:0:0:0:0:1')
        self.assertEqual(proto.getPeer().port, 8080)
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().host, '0:0:0:0:0:0:0:1')
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().port, 8080)
        self.assertEqual(proto.getHost().host, '0:0:0:0:0:0:0:1')
        self.assertEqual(proto.getHost().port, 8888)
        self.assertEqual(proto.wrappedProtocol.transport.getHost().host, '0:0:0:0:0:0:0:1')
        self.assertEqual(proto.wrappedProtocol.transport.getHost().port, 8888)

    def test_validUNIXHeaderResolves_getPeerHost(self) -> None:
        """
        Test if UNIX headers result in the correct host and peer data.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.UNIXAddress(b'/home/test/sockets/server.sock'))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(self.UNIXHEADER)
        self.assertEqual(proto.getPeer().name, b'/home/tests/mysockets/sock')
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().name, b'/home/tests/mysockets/sock')
        self.assertEqual(proto.getHost().name, b'/home/tests/mysockets/sock')
        self.assertEqual(proto.wrappedProtocol.transport.getHost().name, b'/home/tests/mysockets/sock')

    def test_overflowBytesSentToWrappedProtocol(self) -> None:
        """
        Test if non-header bytes are passed to the wrapped protocol.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(self.IPV6HEADER + b'HTTP/1.1 / GET')
        self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET')

    def test_overflowBytesSentToWrappedProtocolChunks(self) -> None:
        """
        Test if header streaming passes extra data appropriately.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(self.IPV6HEADER[:18])
        proto.dataReceived(self.IPV6HEADER[18:] + b'HTTP/1.1 / GET')
        self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET')