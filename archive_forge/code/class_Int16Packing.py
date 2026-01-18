import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class Int16Packing(unittest.TestCase):
    knownValues = ((10, b'\x00\n'), (500, b'\x01\xf4'), (5340, b'\x14\xdc'), (51298, b'\xc8b'), (65535, b'\xff\xff'))

    def test16bitPacking(self):
        """ pack16bit should give known output for known input """
        for i, s in self.knownValues:
            result = DNS.Lib.pack16bit(i)
            self.assertEqual(s, result)

    def test16bitUnpacking(self):
        """ unpack16bit should give known output for known input """
        for i, s in self.knownValues:
            result = DNS.Lib.unpack16bit(s)
            self.assertEqual(i, result)