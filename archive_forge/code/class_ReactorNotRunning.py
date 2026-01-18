import socket
from incremental import Version
from twisted.python import deprecate
class ReactorNotRunning(RuntimeError):
    """
    Error raised when trying to stop a reactor which is not running.
    """