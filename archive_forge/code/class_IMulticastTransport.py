from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IMulticastTransport(Interface):
    """
    Additional functionality for multicast UDP.
    """

    def getOutgoingInterface() -> str:
        """
        Return interface of outgoing multicast packets.
        """

    def setOutgoingInterface(addr: str) -> None:
        """
        Set interface for outgoing multicast packets.

        Returns Deferred of success.
        """

    def getLoopbackMode() -> bool:
        """
        Return if loopback mode is enabled.
        """

    def setLoopbackMode(mode: bool) -> None:
        """
        Set if loopback mode is enabled.
        """

    def getTTL() -> int:
        """
        Get time to live for multicast packets.
        """

    def setTTL(ttl: int) -> None:
        """
        Set time to live on multicast packets.
        """

    def joinGroup(addr: str, interface: str) -> 'Deferred[None]':
        """
        Join a multicast group. Returns L{Deferred} of success or failure.

        If an error occurs, the returned L{Deferred} will fail with
        L{error.MulticastJoinError}.
        """

    def leaveGroup(addr: str, interface: str) -> 'Deferred[None]':
        """
        Leave multicast group, return L{Deferred} of success.
        """