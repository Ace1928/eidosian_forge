import unittest
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.Cipher import ARC2
class KeyLength(unittest.TestCase):

    def runTest(self):
        ARC2.new(b'\x00' * 16, ARC2.MODE_ECB, effective_keylen=40)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 4, ARC2.MODE_ECB)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 129, ARC2.MODE_ECB)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 16, ARC2.MODE_ECB, effective_keylen=39)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 16, ARC2.MODE_ECB, effective_keylen=1025)