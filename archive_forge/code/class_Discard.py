import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol
class Discard(protocol.Protocol):
    """
    Discard any received data (RFC 863).
    """

    def dataReceived(self, data):
        pass