import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Cipher import DES3
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.Util.py3compat import bchr, tostr
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
class DegenerateToDESTest(unittest.TestCase):

    def runTest(self):
        sub_key1 = bchr(1) * 8
        sub_key2 = bchr(255) * 8
        self.assertRaises(ValueError, DES3.new, sub_key1 * 2 + sub_key2, DES3.MODE_ECB)
        self.assertRaises(ValueError, DES3.new, sub_key1 + sub_key2 * 2, DES3.MODE_ECB)
        self.assertRaises(ValueError, DES3.new, sub_key1 * 3, DES3.MODE_ECB)
        self.assertRaises(ValueError, DES3.new, sub_key1 + sub_key2 + strxor_c(sub_key2, 1), DES3.MODE_ECB)