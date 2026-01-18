import socket
from incremental import Version
from twisted.python import deprecate
class MessageLengthError(Exception):
    __doc__ = MESSAGE = 'Message is too long to send'

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.args:
            s = '{}: {}'.format(s, ' '.join(self.args))
        s = '%s.' % s
        return s