import json
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class SivTests(unittest.TestCase):
    key_256 = get_tag_random('key_256', 32)
    key_384 = get_tag_random('key_384', 48)
    key_512 = get_tag_random('key_512', 64)
    nonce_96 = get_tag_random('nonce_128', 12)
    data = get_tag_random('data', 128)

    def test_loopback_128(self):
        for key in (self.key_256, self.key_384, self.key_512):
            cipher = AES.new(key, AES.MODE_SIV, nonce=self.nonce_96)
            pt = get_tag_random('plaintext', 16 * 100)
            ct, mac = cipher.encrypt_and_digest(pt)
            cipher = AES.new(key, AES.MODE_SIV, nonce=self.nonce_96)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)

    def test_nonce(self):
        AES.new(self.key_256, AES.MODE_SIV)
        cipher = AES.new(self.key_256, AES.MODE_SIV, self.nonce_96)
        ct1, tag1 = cipher.encrypt_and_digest(self.data)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct2, tag2 = cipher.encrypt_and_digest(self.data)
        self.assertEqual(ct1 + tag1, ct2 + tag2)

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_256, AES.MODE_SIV, nonce=u'test12345678')

    def test_nonce_length(self):
        self.assertRaises(ValueError, AES.new, self.key_256, AES.MODE_SIV, nonce=b'')
        for x in range(1, 128):
            cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=bchr(1) * x)
            cipher.encrypt_and_digest(b'\x01')

    def test_block_size_128(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_nonce_attribute(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)
        self.assertFalse(hasattr(AES.new(self.key_256, AES.MODE_SIV), 'nonce'))

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_256, AES.MODE_SIV, self.nonce_96, 7)
        self.assertRaises(TypeError, AES.new, self.key_256, AES.MODE_SIV, nonce=self.nonce_96, unknown=7)
        AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96, use_aesni=False)

    def test_encrypt_excludes_decrypt(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.encrypt_and_digest(self.data)
        self.assertRaises(TypeError, cipher.decrypt, self.data)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.encrypt_and_digest(self.data)
        self.assertRaises(TypeError, cipher.decrypt_and_verify, self.data, self.data)

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt_and_verify, u'test1234567890-*', b'xxxx')

    def test_mac_len(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Cryptodome.Util.strxor import strxor_c
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)
        invalid_mac = strxor_c(mac, 1)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct, invalid_mac)

    def test_hex_mac(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_bytearray(self):
        key = bytearray(self.key_256)
        nonce = bytearray(self.nonce_96)
        data = bytearray(self.data)
        header = bytearray(self.data)
        cipher1 = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher1.update(self.data)
        ct, tag = cipher1.encrypt_and_digest(self.data)
        cipher2 = AES.new(key, AES.MODE_SIV, nonce=nonce)
        key[:3] = b'\xff\xff\xff'
        nonce[:3] = b'\xff\xff\xff'
        cipher2.update(header)
        header[:3] = b'\xff\xff\xff'
        ct_test, tag_test = cipher2.encrypt_and_digest(data)
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        key = bytearray(self.key_256)
        nonce = bytearray(self.nonce_96)
        header = bytearray(self.data)
        ct_ba = bytearray(ct)
        tag_ba = bytearray(tag)
        cipher3 = AES.new(key, AES.MODE_SIV, nonce=nonce)
        key[:3] = b'\xff\xff\xff'
        nonce[:3] = b'\xff\xff\xff'
        cipher3.update(header)
        header[:3] = b'\xff\xff\xff'
        pt_test = cipher3.decrypt_and_verify(ct_ba, tag_ba)
        self.assertEqual(self.data, pt_test)

    def test_memoryview(self):
        key = memoryview(bytearray(self.key_256))
        nonce = memoryview(bytearray(self.nonce_96))
        data = memoryview(bytearray(self.data))
        header = memoryview(bytearray(self.data))
        cipher1 = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher1.update(self.data)
        ct, tag = cipher1.encrypt_and_digest(self.data)
        cipher2 = AES.new(key, AES.MODE_SIV, nonce=nonce)
        key[:3] = b'\xff\xff\xff'
        nonce[:3] = b'\xff\xff\xff'
        cipher2.update(header)
        header[:3] = b'\xff\xff\xff'
        ct_test, tag_test = cipher2.encrypt_and_digest(data)
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        key = memoryview(bytearray(self.key_256))
        nonce = memoryview(bytearray(self.nonce_96))
        header = memoryview(bytearray(self.data))
        ct_ba = memoryview(bytearray(ct))
        tag_ba = memoryview(bytearray(tag))
        cipher3 = AES.new(key, AES.MODE_SIV, nonce=nonce)
        key[:3] = b'\xff\xff\xff'
        nonce[:3] = b'\xff\xff\xff'
        cipher3.update(header)
        header[:3] = b'\xff\xff\xff'
        pt_test = cipher3.decrypt_and_verify(ct_ba, tag_ba)
        self.assertEqual(self.data, pt_test)

    def test_output_param(self):
        pt = b'5' * 128
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(pt)
        output = bytearray(128)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        res, tag_out = cipher.encrypt_and_digest(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        self.assertEqual(tag, tag_out)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        res = cipher.decrypt_and_verify(ct, tag, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):
        pt = b'5' * 128
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(pt)
        output = memoryview(bytearray(128))
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.encrypt_and_digest(pt, output=output)
        self.assertEqual(ct, output)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.decrypt_and_verify(ct, tag, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128
        pt = b'5' * LEN_PT
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(pt)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt_and_digest, pt, output=b'0' * LEN_PT)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt_and_verify, ct, tag, output=b'0' * LEN_PT)
        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.encrypt_and_digest, pt, output=shorter_output)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct, tag, output=shorter_output)