from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IUNIXDatagramConnectedTransport(Interface):
    """
    Transport for UDP ConnectedPacketProtocols.
    """

    def write(packet: bytes) -> None:
        """
        Write packet to address we are connected to.
        """

    def getHost() -> 'UNIXAddress':
        """
        Returns L{UNIXAddress}.
        """

    def getPeer() -> 'UNIXAddress':
        """
        Returns L{UNIXAddress}.
        """