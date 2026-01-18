import socket
from incremental import Version
from twisted.python import deprecate
class InvalidAddressError(ValueError):
    """
    An invalid address was specified (i.e. neither IPv4 or IPv6, or expected
    one and got the other).

    @ivar address: See L{__init__}
    @ivar message: See L{__init__}
    """

    def __init__(self, address, message):
        """
        @param address: The address that was provided.
        @type address: L{bytes}
        @param message: A native string of additional information provided by
            the calling context.
        @type address: L{str}
        """
        self.address = address
        self.message = message