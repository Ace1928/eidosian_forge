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
class RRunpackerText(RRunpackerDefault):

    def __init__(self, buf):
        RRunpackerDefault.__init__(self, buf)

    def getAdata(self):
        if DNS.LABEL_UTF8:
            enc = 'utf8'
        else:
            enc = DNS.LABEL_ENCODING
        return self.getaddr().decode(enc)

    def getAAAAdata(self):
        return bin2addr6(self.getaddr6())

    def getTXTdata(self):
        if DNS.LABEL_UTF8:
            enc = 'utf8'
        else:
            enc = DNS.LABEL_ENCODING
        tlist = []
        while self.offset != self.rdend:
            tlist.append(str(self.getstring(), enc))
        return tlist