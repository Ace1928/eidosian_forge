import socket
from incremental import Version
from twisted.python import deprecate
class DNSLookupError(IOError):
    __doc__ = MESSAGE = 'DNS lookup failed'

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.args:
            s = '{}: {}'.format(s, ' '.join(self.args))
        s = '%s.' % s
        return s