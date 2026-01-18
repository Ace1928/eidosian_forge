import unittest
from binascii import hexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import SHA3_256 as SHA3
from Cryptodome.Util.py3compat import b
class APITest(unittest.TestCase):

    def test_update_after_digest(self):
        msg = b('rrrrttt')
        h = SHA3.new(data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = SHA3.new(data=msg).digest()
        h = SHA3.new(data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)