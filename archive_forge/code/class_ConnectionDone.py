import socket
from incremental import Version
from twisted.python import deprecate
class ConnectionDone(ConnectionClosed):
    __doc__ = MESSAGE = 'Connection was closed cleanly'

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.args:
            s = '{}: {}'.format(s, ' '.join(self.args))
        s = '%s.' % s
        return s