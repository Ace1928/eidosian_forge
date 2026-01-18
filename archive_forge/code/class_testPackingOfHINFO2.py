import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfHINFO2(PackerTestCase):
    """addHINFO(self, name, klass, ttl, cpu, os)"""

    def doPack(self, p):
        p.addHINFO('core1.lax.foo.com', DNS.Class.IN, 3600, 'cisco', 'ios')

    def doUnpack(self, u):
        return u.getHINFOdata()
    unpackerExpectedResult = (('core1.lax.foo.com', 13, 1, 3600), ('cisco', 'ios'))
    packerExpectedResult = b'\x05core1\x03lax\x03foo\x03com\x00\x00\r\x00\x01' + b'\x00\x00\x0e\x10\x00\n\x05cisco\x03ios'