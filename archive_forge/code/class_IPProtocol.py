import socket
import struct
from zope.interface import implementer
from twisted.internet import protocol
from twisted.pair import raw
@implementer(raw.IRawPacketProtocol)
class IPProtocol(protocol.AbstractDatagramProtocol):

    def __init__(self):
        self.ipProtos = {}

    def addProto(self, num, proto):
        proto = raw.IRawDatagramProtocol(proto)
        if num < 0:
            raise TypeError('Added protocol must be positive or zero')
        if num >= MAX_SIZE:
            raise TypeError('Added protocol must fit in 32 bits')
        if num not in self.ipProtos:
            self.ipProtos[num] = []
        self.ipProtos[num].append(proto)

    def datagramReceived(self, data, partial, dest, source, protocol):
        header = IPHeader(data)
        for proto in self.ipProtos.get(header.protocol, ()):
            proto.datagramReceived(data=data[20:], partial=partial, source=header.saddr, dest=header.daddr, protocol=header.protocol, version=header.version, ihl=header.ihl, tos=header.tos, tot_len=header.tot_len, fragment_id=header.fragment_id, fragment_offset=header.fragment_offset, dont_fragment=header.dont_fragment, more_fragments=header.more_fragments, ttl=header.ttl)