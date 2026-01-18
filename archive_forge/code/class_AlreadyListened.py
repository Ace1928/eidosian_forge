import socket
from incremental import Version
from twisted.python import deprecate
class AlreadyListened(Exception):
    """
    An attempt was made to listen on a file descriptor which can only be
    listened on once.
    """