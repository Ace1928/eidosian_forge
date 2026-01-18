import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import tostr, tobytes
from Cryptodome.Hash import (HMAC, MD5, SHA1, SHA256,
class HMAC_None(unittest.TestCase):

    def runTest(self):
        key = b'\x04' * 20
        one = HMAC.new(key, b'', SHA1).digest()
        two = HMAC.new(key, None, SHA1).digest()
        self.assertEqual(one, two)