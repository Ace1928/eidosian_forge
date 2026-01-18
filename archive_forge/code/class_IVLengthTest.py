import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
class IVLengthTest(unittest.TestCase):

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.key = b(params['key'])

    def shortDescription(self):
        return 'Check that all modes except MODE_ECB and MODE_CTR require an IV of the proper length'

    def runTest(self):
        self.assertRaises(TypeError, self.module.new, a2b_hex(self.key), self.module.MODE_ECB, b(''))

    def _dummy_counter(self):
        return '\x00' * self.module.block_size