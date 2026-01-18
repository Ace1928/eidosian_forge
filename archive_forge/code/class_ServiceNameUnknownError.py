import socket
from incremental import Version
from twisted.python import deprecate
class ServiceNameUnknownError(ConnectError):
    __doc__ = MESSAGE = 'Service name given as port is unknown'