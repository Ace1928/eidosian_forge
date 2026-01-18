import struct
from zope.interface import Interface, implementer
from twisted.internet import protocol
from twisted.pair import raw
class EthernetHeader:

    def __init__(self, data):
        self.dest, self.source, self.proto = struct.unpack('!6s6sH', data[:6 + 6 + 2])