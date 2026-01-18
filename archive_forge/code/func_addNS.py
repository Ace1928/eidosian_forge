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
def addNS(self, name, klass, ttl, nsdname):
    self.addRRheader(name, Type.NS, klass, ttl)
    self.addname(nsdname)
    self.endRR()