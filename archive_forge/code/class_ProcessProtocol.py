import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
@implementer(interfaces.IProcessProtocol)
class ProcessProtocol(BaseProtocol):
    """
    Base process protocol implementation which does simple dispatching for
    stdin, stdout, and stderr file descriptors.
    """
    transport: Optional[interfaces.IProcessTransport] = None

    def childDataReceived(self, childFD: int, data: bytes) -> None:
        if childFD == 1:
            self.outReceived(data)
        elif childFD == 2:
            self.errReceived(data)

    def outReceived(self, data: bytes) -> None:
        """
        Some data was received from stdout.
        """

    def errReceived(self, data: bytes) -> None:
        """
        Some data was received from stderr.
        """

    def childConnectionLost(self, childFD: int) -> None:
        if childFD == 0:
            self.inConnectionLost()
        elif childFD == 1:
            self.outConnectionLost()
        elif childFD == 2:
            self.errConnectionLost()

    def inConnectionLost(self):
        """
        This will be called when stdin is closed.
        """

    def outConnectionLost(self):
        """
        This will be called when stdout is closed.
        """

    def errConnectionLost(self):
        """
        This will be called when stderr is closed.
        """

    def processExited(self, reason: failure.Failure) -> None:
        """
        This will be called when the subprocess exits.

        @type reason: L{twisted.python.failure.Failure}
        """

    def processEnded(self, reason: failure.Failure) -> None:
        """
        Called when the child process exits and all file descriptors
        associated with it have been closed.

        @type reason: L{twisted.python.failure.Failure}
        """