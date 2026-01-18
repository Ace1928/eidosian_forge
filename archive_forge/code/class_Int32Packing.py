import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class Int32Packing(unittest.TestCase):
    knownValues = ((10, b'\x00\x00\x00\n'), (500, b'\x00\x00\x01\xf4'), (5340, b'\x00\x00\x14\xdc'), (51298, b'\x00\x00\xc8b'), (65535, b'\x00\x00\xff\xff'), (33265535, b'\x01\xfb\x97\x7f'), (147483647, b'\x08\xcak\xff'), (2147483647, b'\x7f\xff\xff\xff'))

    def test32bitPacking(self):
        """ pack32bit should give known output for known input """
        for i, s in self.knownValues:
            result = DNS.Lib.pack32bit(i)
            self.assertEqual(s, result)

    def test32bitUnpacking(self):
        """ unpack32bit should give known output for known input """
        for i, s in self.knownValues:
            result = DNS.Lib.unpack32bit(s)
            self.assertEqual(i, result)