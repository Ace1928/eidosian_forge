import socket
from incremental import Version
from twisted.python import deprecate
class ReactorNotRestartable(RuntimeError):
    """
    Error raised when trying to run a reactor which was stopped.
    """