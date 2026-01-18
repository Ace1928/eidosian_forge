import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfSOA(PackerTestCase):
    """addSOA(self, name, klass, ttl, mname,
           rname, serial, refresh, retry, expire, minimum)"""

    def doPack(self, p):
        p.addSOA('ekit-inc.com', DNS.Class.IN, 3600, 'ns1.ekorp.com', 'hostmaster.ekit-inc.com', 2002020301, 100, 200, 300, 400)

    def doUnpack(self, u):
        return u.getSOAdata()
    unpackerExpectedResult = (('ekit-inc.com', 6, 1, 3600), ('ns1.ekorp.com', 'hostmaster', ('serial', 2002020301), ('refresh ', 100, '1 minutes'), ('retry', 200, '3 minutes'), ('expire', 300, '5 minutes'), ('minimum', 400, '6 minutes')))
    packerExpectedResult = b'\x08ekit-inc\x03com\x00\x00\x06\x00\x01\x00\x00\x0e\x10\x00,\x03ns1\x05ekorp\xc0\t\nhostmaster\x00wTg\xcd\x00\x00\x00d\x00\x00\x00\xc8\x00\x00\x01,\x00\x00\x01\x90'