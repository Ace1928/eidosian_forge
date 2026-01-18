import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfA(PackerTestCase):
    """addA(self, name, klass, ttl, address)"""

    def doPack(self, p):
        p.addA('www02.ekit.com', DNS.Class.IN, 86400, '192.168.10.2')

    def doUnpack(self, u):
        return u.getAdata()
    unpackerExpectedResult = (('www02.ekit.com', 1, 1, 86400), '192.168.10.2')
    packerExpectedResult = b'\x05www02\x04ekit\x03com\x00\x00\x01\x00\x01\x00\x01Q\x80\x00\x04\xc0\xa8\n\x02'