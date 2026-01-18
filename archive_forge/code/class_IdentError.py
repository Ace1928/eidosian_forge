import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
class IdentError(Exception):
    """
    Can't determine connection owner; reason unknown.
    """
    identDescription = 'UNKNOWN-ERROR'

    def __str__(self) -> str:
        return self.identDescription