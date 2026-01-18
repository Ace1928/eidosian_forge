import socket
from incremental import Version
from twisted.python import deprecate
class VerifyError(Exception):
    __doc__ = MESSAGE = 'Could not verify something that was supposed to be signed.'