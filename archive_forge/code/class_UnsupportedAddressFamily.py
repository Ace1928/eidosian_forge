import socket
from incremental import Version
from twisted.python import deprecate
class UnsupportedAddressFamily(Exception):
    """
    An attempt was made to use a socket with an address family (eg I{AF_INET},
    I{AF_INET6}, etc) which is not supported by the reactor.
    """