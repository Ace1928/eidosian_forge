import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfA3(PackerTestCase):
    """addA(self, name, ttl, address)"""

    def doPack(self, p):
        p.addA('www.zol.com', DNS.Class.IN, 86400, '192.168.10.4')
        p.addA('www.zol.com', DNS.Class.IN, 86400, '192.168.10.3')
        p.addA('www.zol.com', DNS.Class.IN, 86400, '192.168.10.2')
        p.addA('www.zol.com', DNS.Class.IN, 86400, '192.168.10.1')

    def doUnpack(self, u):
        u1, d1, u2, d2, u3, d3, u4 = (u.getAdata(), u.getRRheader(), u.getAdata(), u.getRRheader(), u.getAdata(), u.getRRheader(), u.getAdata())
        return (u1, u2, u3, u4)
    unpackerExpectedResult = (('www.zol.com', 1, 1, 86400), ('192.168.10.4', '192.168.10.3', '192.168.10.2', '192.168.10.1'))
    packerExpectedResult = b'\x03www\x03zol\x03com\x00\x00\x01\x00\x01\x00\x01Q\x80\x00\x04\xc0\xa8\n\x04\x00\x00\x01\x00\x01\x00\x01Q\x80\x00\x04\xc0\xa8\n\x03\x00\x00\x01\x00\x01\x00\x01Q\x80\x00\x04\xc0\xa8\n\x02\x00\x00\x01\x00\x01\x00\x01Q\x80\x00\x04\xc0\xa8\n\x01'