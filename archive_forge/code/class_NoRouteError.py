import socket
from incremental import Version
from twisted.python import deprecate
class NoRouteError(ConnectError):
    __doc__ = MESSAGE = 'No route to host'