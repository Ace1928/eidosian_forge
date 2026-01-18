from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IListeningPort(Interface):
    """
    A listening port.
    """

    def startListening() -> None:
        """
        Start listening on this port.

        @raise CannotListenError: If it cannot listen on this port (e.g., it is
                                  a TCP port and it cannot bind to the required
                                  port number).
        """

    def stopListening() -> Optional['Deferred[None]']:
        """
        Stop listening on this port.

        If it does not complete immediately, will return Deferred that fires
        upon completion.
        """

    def getHost() -> IAddress:
        """
        Get the host that this port is listening for.

        @return: An L{IAddress} provider.
        """