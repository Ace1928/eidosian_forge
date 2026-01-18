import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfAAAAInteger(PackerTestCase):
    """addAAAA(self, name, klass, ttl, address)"""

    def setUp(self):
        self.RRpacker = DNS.Lib.RRpacker
        self.RRunpacker = DNS.Lib.RRunpackerInteger

    def doPack(self, p):
        addAAAA(p, 'google.com', DNS.Class.IN, 4, '2607:f8b0:4005:802::1005')

    def doUnpack(self, u):
        r = u.getAAAAdata()
        return r
    packerExpectedResult = b'\x06google\x03com\x00\x00\x1c\x00\x01\x00\x00\x00\x04\x00\x10&\x07\xf8\xb0@\x05\x08\x02\x00\x00\x00\x00\x00\x00\x10\x05'
    unpackerExpectedResult = (('google.com', DNS.Type.AAAA, DNS.Class.IN, 4), 50552053919387978162022445795852161029)