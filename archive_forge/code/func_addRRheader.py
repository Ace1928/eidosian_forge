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
def addRRheader(self, name, RRtype, klass, ttl, *rest):
    self.addname(name)
    self.add16bit(RRtype)
    self.add16bit(klass)
    self.add32bit(ttl)
    if rest:
        if rest[1:]:
            raise TypeError('too many args')
        rdlength = rest[0]
    else:
        rdlength = 0
    self.add16bit(rdlength)
    self.rdstart = len(self.buf)