from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorMulticast(Interface):
    """
    UDP socket methods that support multicast.

    IMPORTANT: This is an experimental new interface. It may change
    without backwards compatibility. Suggestions are welcome.
    """

    def listenMulticast(port: int, protocol: 'DatagramProtocol', interface: str, maxPacketSize: int, listenMultiple: bool) -> 'IListeningPort':
        """
        Connects a given
        L{DatagramProtocol<twisted.internet.protocol.DatagramProtocol>} to the
        given numeric UDP port.

        @param listenMultiple: If set to True, allows multiple sockets to
            bind to the same address and port number at the same time.

        @returns: An object which provides L{IListeningPort}.

        @see: L{twisted.internet.interfaces.IMulticastTransport}
        @see: U{http://twistedmatrix.com/documents/current/core/howto/udp.html}
        """