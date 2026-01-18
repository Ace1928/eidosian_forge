from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IProtocolFactory(Interface):
    """
    Interface for protocol factories.
    """

    def buildProtocol(addr: IAddress) -> Optional[IProtocol]:
        """
        Called when a connection has been established to addr.

        If None is returned, the connection is assumed to have been refused,
        and the Port will close the connection.

        @param addr: The address of the newly-established connection

        @return: None if the connection was refused, otherwise an object
                 providing L{IProtocol}.
        """

    def doStart() -> None:
        """
        Called every time this is connected to a Port or Connector.
        """

    def doStop() -> None:
        """
        Called every time this is unconnected from a Port or Connector.
        """