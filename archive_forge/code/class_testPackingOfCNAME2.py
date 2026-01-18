import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfCNAME2(PackerTestCase):
    """addCNAME(self, name, klass, ttl, cname)"""

    def doPack(self, p):
        p.addCNAME('www.cust.com', DNS.Class.IN, 200, 'www023.big.isp.com')

    def doUnpack(self, u):
        return u.getCNAMEdata()
    unpackerExpectedResult = (('www.cust.com', DNS.Type.CNAME, DNS.Class.IN, 200), 'www023.big.isp.com')
    packerExpectedResult = b'\x03www\x04cust\x03com\x00\x00\x05\x00\x01\x00' + b'\x00\x00\xc8\x00\x11\x06www023\x03big\x03isp\xc0\t'