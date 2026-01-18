import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class PackerClassPacking(unittest.TestCase):
    knownPackValues = [(['www.ekit.com'], b'\x03www\x04ekit\x03com\x00'), (['ns1.ekorp.com', 'ns2.ekorp.com', 'ns3.ekorp.com'], b'\x03ns1\x05ekorp\x03com\x00\x03ns2\xc0\x04\x03ns3\xc0\x04'), (['a.root-servers.net.', 'b.root-servers.net.', 'c.root-servers.net.', 'd.root-servers.net.', 'e.root-servers.net.', 'f.root-servers.net.'], b'\x01a\x0croot-servers\x03net\x00\x01b\xc0\x02\x01c\xc0' + b'\x02\x01d\xc0\x02\x01e\xc0\x02\x01f\xc0\x02')]
    knownUnpackValues = [(['www.ekit.com'], b'\x03www\x04ekit\x03com\x00'), (['ns1.ekorp.com', 'ns2.ekorp.com', 'ns3.ekorp.com'], b'\x03ns1\x05ekorp\x03com\x00\x03ns2\xc0\x04\x03ns3\xc0\x04'), (['a.root-servers.net', 'b.root-servers.net', 'c.root-servers.net', 'd.root-servers.net', 'e.root-servers.net', 'f.root-servers.net'], b'\x01a\x0croot-servers\x03net\x00\x01b\xc0\x02\x01c\xc0' + b'\x02\x01d\xc0\x02\x01e\xc0\x02\x01f\xc0\x02')]

    def testPackNames(self):
        from DNS.Lib import Packer
        for namelist, result in self.knownPackValues:
            p = Packer()
            for n in namelist:
                p.addname(n)
            self.assertEqual(p.getbuf(), result)

    def testUnpackNames(self):
        from DNS.Lib import Unpacker
        for namelist, result in self.knownUnpackValues:
            u = Unpacker(result)
            names = []
            for i in range(len(namelist)):
                n = u.getname()
                names.append(n)
            self.assertEqual(names, namelist)