from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IConnector(Interface):
    """
    Object used to interface between connections and protocols.

    Each L{IConnector} manages one connection.
    """

    def stopConnecting() -> None:
        """
        Stop attempting to connect.
        """

    def disconnect() -> None:
        """
        Disconnect regardless of the connection state.

        If we are connected, disconnect, if we are trying to connect,
        stop trying.
        """

    def connect() -> None:
        """
        Try to connect to remote address.
        """

    def getDestination() -> IAddress:
        """
        Return destination this will try to connect to.

        @return: An object which provides L{IAddress}.
        """