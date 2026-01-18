import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
class NoUser(IdentError):
    """
    The connection specified by the port pair is not currently in use or
    currently not owned by an identifiable entity.
    """
    identDescription = 'NO-USER'