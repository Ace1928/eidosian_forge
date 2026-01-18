import socket
from incremental import Version
from twisted.python import deprecate
class ConnectionRefusedError(ConnectError):
    __doc__ = MESSAGE = 'Connection was refused by other side'