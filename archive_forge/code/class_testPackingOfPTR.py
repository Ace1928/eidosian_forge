import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfPTR(PackerTestCase):
    """addPTR(self, name, klass, ttl, ptrdname)"""

    def doPack(self, p):
        p.addPTR('www.ekit-inc.com', DNS.Class.IN, 3600, 'www-real01.ekorp.com')

    def doUnpack(self, u):
        return u.getPTRdata()
    unpackerExpectedResult = (('www.ekit-inc.com', 12, 1, 3600), 'www-real01.ekorp.com')
    packerExpectedResult = b'\x03www\x08ekit-inc\x03com\x00\x00\x0c\x00\x01\x00\x00\x0e\x10\x00\x13\nwww-real01\x05ekorp\xc0\r'