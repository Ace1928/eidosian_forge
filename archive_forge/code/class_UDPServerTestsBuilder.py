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
class UDPServerTestsBuilder(ReactorBuilder, UDPPortTestsMixin, DatagramTransportTestsMixin):
    """
    Run L{UDPPortTestsMixin} tests using newly created UDP
    sockets.
    """
    requiredInterfaces = (IReactorUDP,)

    def getListeningPort(self, reactor, protocol, port=0, interface='', maxPacketSize=8192):
        """
        Get a UDP port from a reactor.

        @param reactor: A reactor used to build the returned
            L{IListeningPort} provider.
        @type reactor: L{twisted.internet.interfaces.IReactorUDP}

        @see: L{twisted.internet.IReactorUDP.listenUDP} for other
            argument and return types.
        """
        return reactor.listenUDP(port, protocol, interface=interface, maxPacketSize=maxPacketSize)