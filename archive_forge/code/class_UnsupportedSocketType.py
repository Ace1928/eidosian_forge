import socket
from incremental import Version
from twisted.python import deprecate
class UnsupportedSocketType(Exception):
    """
    An attempt was made to use a socket of a type (eg I{SOCK_STREAM},
    I{SOCK_DGRAM}, etc) which is not supported by the reactor.
    """