import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
class CfbTests(BlockChainingTests):
    aes_mode = AES.MODE_CFB
    des3_mode = DES3.MODE_CFB

    def test_unaligned_data_128(self):
        plaintexts = [b'7777777'] * 100
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=8)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=8)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=128)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=128)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [b'7777777'] * 100
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=8)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=8)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=64)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=64)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))

    def test_segment_size_128(self):
        for bits in range(8, 129, 8):
            cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=bits)
        for bits in (0, 7, 9, 127, 129):
            self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CFB, self.iv_128, segment_size=bits)

    def test_segment_size_64(self):
        for bits in range(8, 65, 8):
            cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=bits)
        for bits in (0, 7, 9, 63, 65):
            self.assertRaises(ValueError, DES3.new, self.key_192, AES.MODE_CFB, self.iv_64, segment_size=bits)