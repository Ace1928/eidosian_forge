import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
class MemoryViewTest(unittest.TestCase):

    def __init__(self, module, extra_params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.extra_params = extra_params

    def runTest(self):
        data = b'\x00\x01\x02'

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))
        for get_mv in (get_mv_ro, get_mv_rw):
            mv = get_mv(data)
            h1 = self.module.new(data, **self.extra_params)
            h2 = self.module.new(mv, **self.extra_params)
            if not mv.readonly:
                mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())
            mv = get_mv(data)
            h1 = self.module.new(**self.extra_params)
            h2 = self.module.new(**self.extra_params)
            h1.update(data)
            h2.update(mv)
            if not mv.readonly:
                mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())