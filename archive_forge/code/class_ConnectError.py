import socket
from incremental import Version
from twisted.python import deprecate
class ConnectError(Exception):
    __doc__ = MESSAGE = 'An error occurred while connecting'

    def __init__(self, osError=None, string=''):
        self.osError = osError
        Exception.__init__(self, string)

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.osError:
            s = f'{s}: {self.osError}'
        if self.args[0]:
            s = f'{s}: {self.args[0]}'
        s = '%s.' % s
        return s