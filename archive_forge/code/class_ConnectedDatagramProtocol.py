import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
class ConnectedDatagramProtocol(DatagramProtocol):
    """
    Protocol for connected datagram-oriented transport.

    No longer necessary for UDP.
    """

    def datagramReceived(self, datagram):
        """
        Called when a datagram is received.

        @param datagram: the string received from the transport.
        """

    def connectionFailed(self, failure: failure.Failure) -> None:
        """
        Called if connecting failed.

        Usually this will be due to a DNS lookup failure.
        """