import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
class NoDefaultECBTest(unittest.TestCase):

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.key = b(params['key'])

    def runTest(self):
        self.assertRaises(TypeError, self.module.new, a2b_hex(self.key))