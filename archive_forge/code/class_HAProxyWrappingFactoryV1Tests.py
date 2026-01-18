from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
class HAProxyWrappingFactoryV1Tests(unittest.TestCase):
    """
    Test L{twisted.protocols.haproxy.HAProxyWrappingFactory} with v1 PROXY
    headers.
    """

    def test_invalidHeaderDisconnects(self) -> None:
        """
        Test if invalid headers result in connectionLost events.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv4Address('TCP', '127.1.1.1', 8080))
        transport = StringTransportWithDisconnection()
        transport.protocol = proto
        proto.makeConnection(transport)
        proto.dataReceived(b'NOTPROXY anything can go here\r\n')
        self.assertFalse(transport.connected)

    def test_invalidPartialHeaderDisconnects(self) -> None:
        """
        Test if invalid headers result in connectionLost events.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv4Address('TCP', '127.1.1.1', 8080))
        transport = StringTransportWithDisconnection()
        transport.protocol = proto
        proto.makeConnection(transport)
        proto.dataReceived(b'PROXY TCP4 1.1.1.1\r\n')
        proto.dataReceived(b'2.2.2.2 8080\r\n')
        self.assertFalse(transport.connected)

    def test_preDataReceived_getPeerHost(self) -> None:
        """
        Before any data is received the HAProxy protocol will return the same peer
        and host as the IP connection.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 8080))
        transport = StringTransportWithDisconnection(hostAddress=mock.sentinel.host_address, peerAddress=mock.sentinel.peer_address)
        proto.makeConnection(transport)
        self.assertEqual(proto.getHost(), mock.sentinel.host_address)
        self.assertEqual(proto.getPeer(), mock.sentinel.peer_address)

    def test_validIPv4HeaderResolves_getPeerHost(self) -> None:
        """
        Test if IPv4 headers result in the correct host and peer data.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(b'PROXY TCP4 1.1.1.1 2.2.2.2 8080 8888\r\n')
        self.assertEqual(proto.getPeer().host, '1.1.1.1')
        self.assertEqual(proto.getPeer().port, 8080)
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().host, '1.1.1.1')
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().port, 8080)
        self.assertEqual(proto.getHost().host, '2.2.2.2')
        self.assertEqual(proto.getHost().port, 8888)
        self.assertEqual(proto.wrappedProtocol.transport.getHost().host, '2.2.2.2')
        self.assertEqual(proto.wrappedProtocol.transport.getHost().port, 8888)

    def test_validIPv6HeaderResolves_getPeerHost(self) -> None:
        """
        Test if IPv6 headers result in the correct host and peer data.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(b'PROXY TCP6 ::1 ::2 8080 8888\r\n')
        self.assertEqual(proto.getPeer().host, '::1')
        self.assertEqual(proto.getPeer().port, 8080)
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().host, '::1')
        self.assertEqual(proto.wrappedProtocol.transport.getPeer().port, 8080)
        self.assertEqual(proto.getHost().host, '::2')
        self.assertEqual(proto.getHost().port, 8888)
        self.assertEqual(proto.wrappedProtocol.transport.getHost().host, '::2')
        self.assertEqual(proto.wrappedProtocol.transport.getHost().port, 8888)

    def test_overflowBytesSentToWrappedProtocol(self) -> None:
        """
        Test if non-header bytes are passed to the wrapped protocol.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(b'PROXY TCP6 ::1 ::2 8080 8888\r\nHTTP/1.1 / GET')
        self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET')

    def test_overflowBytesSentToWrappedProtocolChunks(self) -> None:
        """
        Test if header streaming passes extra data appropriately.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(b'PROXY TCP6 ::1 ::2 ')
        proto.dataReceived(b'8080 8888\r\nHTTP/1.1 / GET')
        self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET')

    def test_overflowBytesSentToWrappedProtocolAfter(self) -> None:
        """
        Test if wrapper writes all data to wrapped protocol after parsing.
        """
        factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
        proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
        transport = StringTransportWithDisconnection()
        proto.makeConnection(transport)
        proto.dataReceived(b'PROXY TCP6 ::1 ::2 ')
        proto.dataReceived(b'8080 8888\r\nHTTP/1.1 / GET')
        proto.dataReceived(b'\r\n\r\n')
        self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET\r\n\r\n')