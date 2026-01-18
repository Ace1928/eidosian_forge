import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
class OpenPGPTests(BlockChainingTests):
    aes_mode = AES.MODE_OPENPGP
    des3_mode = DES3.MODE_OPENPGP
    key_128 = get_tag_random('key_128', 16)
    key_192 = get_tag_random('key_192', 24)
    iv_128 = get_tag_random('iv_128', 16)
    iv_64 = get_tag_random('iv_64', 8)
    data_128 = get_tag_random('data_128', 16)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        pt = get_tag_random('plaintext', 16 * 100)
        ct = cipher.encrypt(pt)
        eiv, ct = (ct[:18], ct[18:])
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_loopback_64(self):
        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, self.iv_64)
        pt = get_tag_random('plaintext', 8 * 100)
        ct = cipher.encrypt(pt)
        eiv, ct = (ct[:10], ct[10:])
        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, eiv)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_IV_iv_attributes(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        eiv = cipher.encrypt(b'')
        self.assertEqual(cipher.iv, self.iv_128)
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        self.assertEqual(cipher.iv, self.iv_128)

    def test_null_encryption_decryption(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        eiv = cipher.encrypt(b'')
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        self.assertEqual(cipher.decrypt(b''), b'')

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        eiv = cipher.encrypt(b'')
        self.assertRaises(TypeError, cipher.decrypt, b'')
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        cipher.decrypt(b'')
        self.assertRaises(TypeError, cipher.encrypt, b'')

    def test_unaligned_data_128(self):
        plaintexts = [b'7777777'] * 100
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [b'7777777'] * 100
        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, self.iv_64)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, self.iv_64)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))

    def test_output_param(self):
        pass

    def test_output_param_same_buffer(self):
        pass

    def test_output_param_memoryview(self):
        pass

    def test_output_param_neg(self):
        pass