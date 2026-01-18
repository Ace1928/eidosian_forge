import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfMX2(PackerTestCase):
    """addMX(self, name, klass, ttl, preference, exchange)"""

    def doPack(self, p):
        p.addMX('ekit-inc.com.', DNS.Class.IN, 86400, 10, 'mx1.ekorp.com')
        p.addMX('ekit-inc.com.', DNS.Class.IN, 86400, 20, 'mx2.ekorp.com')
        p.addMX('ekit-inc.com.', DNS.Class.IN, 86400, 30, 'mx3.ekorp.com')

    def doUnpack(self, u):
        res = [u.getMXdata()]
        dummy = u.getRRheader()[:4]
        res += u.getMXdata()
        dummy = u.getRRheader()[:4]
        res += u.getMXdata()
        return res
    unpackerExpectedResult = (('ekit-inc.com', 15, 1, 86400), [(10, 'mx1.ekorp.com'), 20, 'mx2.ekorp.com', 30, 'mx3.ekorp.com'])
    packerExpectedResult = b'\x08ekit-inc\x03com\x00\x00\x0f\x00\x01\x00\x01Q\x80\x00' + b'\x0e\x00\n\x03mx1\x05ekorp\xc0\t\x00\x00\x0f\x00\x01\x00' + b'\x01Q\x80\x00\x08\x00\x14\x03mx2\xc0\x1e\x00\x00\x0f\x00' + b'\x01\x00\x01Q\x80\x00\x08\x00\x1e\x03mx3\xc0\x1e'