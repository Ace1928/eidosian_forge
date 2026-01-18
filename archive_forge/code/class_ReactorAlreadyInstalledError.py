import socket
from incremental import Version
from twisted.python import deprecate
class ReactorAlreadyInstalledError(AssertionError):
    """
    Could not install reactor because one is already installed.
    """