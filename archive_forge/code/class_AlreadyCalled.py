import socket
from incremental import Version
from twisted.python import deprecate
class AlreadyCalled(ValueError):
    __doc__ = MESSAGE = 'Tried to cancel an already-called event'

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.args:
            s = '{}: {}'.format(s, ' '.join(self.args))
        s = '%s.' % s
        return s