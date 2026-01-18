import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
class InvalidPort(IdentError):
    """
    Either the local or foreign port was improperly specified. This should
    be returned if either or both of the port ids were out of range (TCP
    port numbers are from 1-65535), negative integers, reals or in any
    fashion not recognized as a non-negative integer.
    """
    identDescription = 'INVALID-PORT'