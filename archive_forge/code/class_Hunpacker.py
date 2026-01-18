import types
import socket
from . import Type
from . import Class
from . import Opcode
from . import Status
import DNS
from .Base import DNSError
from struct import pack as struct_pack
from struct import unpack as struct_unpack
from socket import inet_ntoa, inet_aton, inet_ntop, AF_INET6
class Hunpacker(Unpacker):

    def getHeader(self):
        id = self.get16bit()
        flags = self.get16bit()
        qr, opcode, aa, tc, rd, ra, z, rcode = (flags >> 15 & 1, flags >> 11 & 15, flags >> 10 & 1, flags >> 9 & 1, flags >> 8 & 1, flags >> 7 & 1, flags >> 4 & 7, flags >> 0 & 15)
        qdcount = self.get16bit()
        ancount = self.get16bit()
        nscount = self.get16bit()
        arcount = self.get16bit()
        return (id, qr, opcode, aa, tc, rd, ra, z, rcode, qdcount, ancount, nscount, arcount)