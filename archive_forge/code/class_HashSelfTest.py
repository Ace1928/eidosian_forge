import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
class HashSelfTest(unittest.TestCase):

    def __init__(self, hashmod, description, expected, input, extra_params):
        unittest.TestCase.__init__(self)
        self.hashmod = hashmod
        self.expected = expected.lower()
        self.input = input
        self.description = description
        self.extra_params = extra_params

    def shortDescription(self):
        return self.description

    def runTest(self):
        h = self.hashmod.new(**self.extra_params)
        h.update(self.input)
        out1 = binascii.b2a_hex(h.digest())
        out2 = h.hexdigest()
        h = self.hashmod.new(self.input, **self.extra_params)
        out3 = h.hexdigest()
        out4 = binascii.b2a_hex(h.digest())
        self.assertEqual(self.expected, out1)
        if sys.version_info[0] == 2:
            self.assertEqual(self.expected, out2)
            self.assertEqual(self.expected, out3)
        else:
            self.assertEqual(self.expected.decode(), out2)
            self.assertEqual(self.expected.decode(), out3)
        self.assertEqual(self.expected, out4)
        if self.hashmod.__name__ not in ('Cryptodome.Hash.MD5', 'Cryptodome.Hash.SHA1') or hasattr(h, 'new'):
            h2 = h.new()
            h2.update(self.input)
            out5 = binascii.b2a_hex(h2.digest())
            self.assertEqual(self.expected, out5)