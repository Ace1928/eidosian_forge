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
class UDPFDServerTestsBuilder(ReactorBuilder, UDPPortTestsMixin, DatagramTransportTestsMixin):
    """
    Run L{UDPPortTestsMixin} tests using adopted UDP sockets.
    """
    requiredInterfaces = (IReactorSocket,)

    def getListeningPort(self, reactor, protocol, port=0, interface='', maxPacketSize=8192):
        """
        Get a UDP port from a reactor, wrapping an already-initialized file
        descriptor.

        @param reactor: A reactor used to build the returned
            L{IListeningPort} provider.
        @type reactor: L{twisted.internet.interfaces.IReactorSocket}

        @param port: A port number to which the adopted socket will be
            bound.
        @type port: C{int}

        @param interface: The local IPv4 or IPv6 address to which the
            adopted socket will be bound.  defaults to '', ie all IPv4
            addresses.
        @type interface: C{str}

        @see: L{twisted.internet.IReactorSocket.adoptDatagramPort} for other
            argument and return types.
        """
        if IReactorSocket.providedBy(reactor):
            if ':' in interface:
                domain = socket.AF_INET6
                address = socket.getaddrinfo(interface, port)[0][4]
            else:
                domain = socket.AF_INET
                address = (interface, port)
            portSock = socket.socket(domain, socket.SOCK_DGRAM)
            portSock.bind(address)
            portSock.setblocking(False)
            try:
                return reactor.adoptDatagramPort(portSock.fileno(), portSock.family, protocol, maxPacketSize)
            finally:
                portSock.fileno()
                portSock.close()
        else:
            raise SkipTest('Reactor does not provide IReactorSocket')