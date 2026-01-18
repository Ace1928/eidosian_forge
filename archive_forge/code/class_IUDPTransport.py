from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IUDPTransport(Interface):
    """
    Transport for UDP DatagramProtocols.
    """

    def write(packet: bytes, addr: Optional[Tuple[str, int]]) -> None:
        """
        Write packet to given address.

        @param addr: a tuple of (ip, port). For connected transports must
                     be the address the transport is connected to, or None.
                     In non-connected mode this is mandatory.

        @raise twisted.internet.error.MessageLengthError: C{packet} was too
        long.
        """

    def connect(host: str, port: int) -> None:
        """
        Connect the transport to an address.

        This changes it to connected mode. Datagrams can only be sent to
        this address, and will only be received from this address. In addition
        the protocol's connectionRefused method might get called if destination
        is not receiving datagrams.

        @param host: an IP address, not a domain name ('127.0.0.1', not 'localhost')
        @param port: port to connect to.
        """

    def getHost() -> Union['IPv4Address', 'IPv6Address']:
        """
        Get this port's host address.

        @return: an address describing the listening port.
        """

    def stopListening() -> Optional['Deferred[None]']:
        """
        Stop listening on this port.

        If it does not complete immediately, will return L{Deferred} that fires
        upon completion.
        """

    def setBroadcastAllowed(enabled: bool) -> None:
        """
        Set whether this port may broadcast.

        @param enabled: Whether the port may broadcast.
        """

    def getBroadcastAllowed() -> bool:
        """
        Checks if broadcast is currently allowed on this port.

        @return: Whether this port may broadcast.
        """