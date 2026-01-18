import socket
import struct
from zope.interface import implementer
from twisted.internet import protocol
from twisted.pair import raw
class IPHeader:

    def __init__(self, data):
        ihlversion, self.tos, self.tot_len, self.fragment_id, frag_off, self.ttl, self.protocol, self.check, saddr, daddr = struct.unpack('!BBHHHBBH4s4s', data[:20])
        self.saddr = socket.inet_ntoa(saddr)
        self.daddr = socket.inet_ntoa(daddr)
        self.version = ihlversion & 15
        self.ihl = (ihlversion & 240) >> 4 << 2
        self.fragment_offset = frag_off & 8191
        self.dont_fragment = frag_off & 16384 != 0
        self.more_fragments = frag_off & 8192 != 0