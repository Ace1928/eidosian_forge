import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
class HiddenUser(IdentError):
    """
    The server was able to identify the user of this port, but the
    information was not returned at the request of the user.
    """
    identDescription = 'HIDDEN-USER'