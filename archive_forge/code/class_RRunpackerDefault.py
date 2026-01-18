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
class RRunpackerDefault(RRunpacker):

    def __init__(self, buf):
        RRunpacker.__init__(self, buf)
        self.rdend = None

    def getAdata(self):
        if DNS.LABEL_UTF8:
            enc = 'utf8'
        else:
            enc = DNS.LABEL_ENCODING
        x = socket.inet_aton(self.getaddr().decode(enc))
        return ipaddress.IPv4Address(struct_unpack('!I', x)[0])

    def getAAAAdata(self):
        return ipaddress.IPv6Address(bin2addr6(self.getaddr6()))