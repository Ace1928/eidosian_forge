import socket
from incremental import Version
from twisted.python import deprecate
class ConnectingCancelledError(Exception):
    """
    An C{Exception} that will be raised when an L{IStreamClientEndpoint} is
    cancelled before it connects.

    @ivar address: The L{IAddress} that is the destination of the
        cancelled L{IStreamClientEndpoint}.
    """

    def __init__(self, address):
        """
        @param address: The L{IAddress} that is the destination of the
            L{IStreamClientEndpoint} that was cancelled.
        """
        Exception.__init__(self, address)
        self.address = address