import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
class HashTestOID(unittest.TestCase):

    def __init__(self, hashmod, oid, extra_params):
        unittest.TestCase.__init__(self)
        self.hashmod = hashmod
        self.oid = oid
        self.extra_params = extra_params

    def runTest(self):
        h = self.hashmod.new(**self.extra_params)
        self.assertEqual(h.oid, self.oid)