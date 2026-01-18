import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfMX(PackerTestCase):
    """addMX(self, name, klass, ttl, preference, exchange)"""

    def doPack(self, p):
        p.addMX('sub.domain.com', DNS.Class.IN, 86400, 10, 'mailhost1.isp.com')

    def doUnpack(self, u):
        return u.getMXdata()
    packerExpectedResult = b'\x03sub\x06domain\x03com\x00\x00\x0f\x00\x01' + b'\x00\x01Q\x80\x00\x12\x00\n\tmailhost1\x03isp\xc0\x0b'
    unpackerExpectedResult = (('sub.domain.com', 15, 1, 86400), (10, 'mailhost1.isp.com'))