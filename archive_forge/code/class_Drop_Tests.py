import unittest
from Cryptodome.Util.py3compat import b
from Cryptodome.SelfTest.st_common import list_test_cases
from binascii import unhexlify
from Cryptodome.Cipher import ARC4
class Drop_Tests(unittest.TestCase):
    key = b('ª') * 16
    data = b('\x00') * 5000

    def setUp(self):
        self.cipher = ARC4.new(self.key)

    def test_drop256_encrypt(self):
        cipher_drop = ARC4.new(self.key, 256)
        ct_drop = cipher_drop.encrypt(self.data[:16])
        ct = self.cipher.encrypt(self.data)[256:256 + 16]
        self.assertEqual(ct_drop, ct)

    def test_drop256_decrypt(self):
        cipher_drop = ARC4.new(self.key, 256)
        pt_drop = cipher_drop.decrypt(self.data[:16])
        pt = self.cipher.decrypt(self.data)[256:256 + 16]
        self.assertEqual(pt_drop, pt)