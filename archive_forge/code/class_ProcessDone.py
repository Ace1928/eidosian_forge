import socket
from incremental import Version
from twisted.python import deprecate
class ProcessDone(ConnectionDone):
    __doc__ = MESSAGE = 'A process has ended without apparent errors'

    def __init__(self, status):
        Exception.__init__(self, 'process finished with exit code 0')
        self.exitCode = 0
        self.signal = None
        self.status = status