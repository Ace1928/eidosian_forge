import struct
from zope.interface import implementer
from twisted.internet import protocol
from twisted.pair import raw
class UDPHeader:

    def __init__(self, data):
        self.source, self.dest, self.len, self.check = struct.unpack('!HHHH', data[:8])