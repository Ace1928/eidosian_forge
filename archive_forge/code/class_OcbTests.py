import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
class OcbTests(unittest.TestCase):
    key_128 = get_tag_random('key_128', 16)
    nonce_96 = get_tag_random('nonce_128', 12)
    data = get_tag_random('data', 128)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        pt = get_tag_random('plaintext', 16 * 100)
        ct, mac = cipher.encrypt_and_digest(pt)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        pt2 = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(pt, pt2)

    def test_nonce(self):
        AES.new(self.key_128, AES.MODE_OCB)
        cipher = AES.new(self.key_128, AES.MODE_OCB, self.nonce_96)
        ct = cipher.encrypt(self.data)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertEqual(ct, cipher.encrypt(self.data))

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_OCB, nonce=u'test12345678')

    def test_nonce_length(self):
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB, nonce=b(''))
        for length in range(1, 16):
            AES.new(self.key_128, AES.MODE_OCB, nonce=self.data[:length])
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB, nonce=self.data)

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertEqual(cipher.block_size, AES.block_size)
        nonce1 = AES.new(self.key_128, AES.MODE_OCB).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_OCB).nonce
        self.assertEqual(len(nonce1), 15)
        self.assertNotEqual(nonce1, nonce2)

    def test_nonce_attribute(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)
        nonce1 = AES.new(self.key_128, AES.MODE_OCB).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_OCB).nonce
        self.assertEqual(len(nonce1), 15)
        self.assertNotEqual(nonce1, nonce2)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_OCB, self.nonce_96, 7)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_OCB, nonce=self.nonce_96, unknown=7)
        AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96, use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in ('encrypt', 'decrypt'):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            result = getattr(cipher, func)(b(''))
            self.assertEqual(result, b(''))

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.encrypt(b('xyz'))
        self.assertRaises(TypeError, cipher.decrypt, b('xyz'))
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.decrypt(b('xyz'))
        self.assertRaises(TypeError, cipher.encrypt, b('xyz'))

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_mac_len(self):
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB, nonce=self.nonce_96, mac_len=7)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB, nonce=self.nonce_96, mac_len=16 + 1)
        for mac_len in range(8, 16 + 1):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96, mac_len=mac_len)
            _, mac = cipher.encrypt_and_digest(self.data)
            self.assertEqual(len(mac), mac_len)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Cryptodome.Util.strxor import strxor_c
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)
        invalid_mac = strxor_c(mac, 1)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct, invalid_mac)

    def test_hex_mac(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_message_chunks(self):
        auth_data = get_tag_random('authenticated data', 127)
        plaintext = get_tag_random('plaintext', 127)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.update(auth_data)
        ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

        def break_up(data, chunk_length):
            return [data[i:i + chunk_length] for i in range(0, len(data), chunk_length)]
        for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            pt2 = b('')
            for chunk in break_up(ciphertext, chunk_length):
                pt2 += cipher.decrypt(chunk)
            pt2 += cipher.decrypt()
            self.assertEqual(plaintext, pt2)
            cipher.verify(ref_mac)
        for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            ct2 = b('')
            for chunk in break_up(plaintext, chunk_length):
                ct2 += cipher.encrypt(chunk)
            ct2 += cipher.encrypt()
            self.assertEqual(ciphertext, ct2)
            self.assertEqual(cipher.digest(), ref_mac)

    def test_bytearray(self):
        key_ba = bytearray(self.key_128)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data)
        data_ba = bytearray(self.data)
        cipher1 = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data) + cipher1.encrypt()
        tag = cipher1.digest()
        cipher2 = AES.new(key_ba, AES.MODE_OCB, nonce=nonce_ba)
        key_ba[:3] = b'\xff\xff\xff'
        nonce_ba[:3] = b'\xff\xff\xff'
        cipher2.update(header_ba)
        header_ba[:3] = b'\xff\xff\xff'
        ct_test = cipher2.encrypt(data_ba) + cipher2.encrypt()
        data_ba[:3] = b'\xff\xff\xff'
        tag_test = cipher2.digest()
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        key_ba = bytearray(self.key_128)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data)
        del data_ba
        cipher4 = AES.new(key_ba, AES.MODE_OCB, nonce=nonce_ba)
        key_ba[:3] = b'\xff\xff\xff'
        nonce_ba[:3] = b'\xff\xff\xff'
        cipher4.update(header_ba)
        header_ba[:3] = b'\xff\xff\xff'
        pt_test = cipher4.decrypt_and_verify(bytearray(ct_test), bytearray(tag_test))
        self.assertEqual(self.data, pt_test)

    def test_memoryview(self):
        key_mv = memoryview(bytearray(self.key_128))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data))
        data_mv = memoryview(bytearray(self.data))
        cipher1 = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data) + cipher1.encrypt()
        tag = cipher1.digest()
        cipher2 = AES.new(key_mv, AES.MODE_OCB, nonce=nonce_mv)
        key_mv[:3] = b'\xff\xff\xff'
        nonce_mv[:3] = b'\xff\xff\xff'
        cipher2.update(header_mv)
        header_mv[:3] = b'\xff\xff\xff'
        ct_test = cipher2.encrypt(data_mv) + cipher2.encrypt()
        data_mv[:3] = b'\xff\xff\xff'
        tag_test = cipher2.digest()
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        key_mv = memoryview(bytearray(self.key_128))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data))
        del data_mv
        cipher4 = AES.new(key_mv, AES.MODE_OCB, nonce=nonce_mv)
        key_mv[:3] = b'\xff\xff\xff'
        nonce_mv[:3] = b'\xff\xff\xff'
        cipher4.update(header_mv)
        header_mv[:3] = b'\xff\xff\xff'
        pt_test = cipher4.decrypt_and_verify(memoryview(ct_test), memoryview(tag_test))
        self.assertEqual(self.data, pt_test)