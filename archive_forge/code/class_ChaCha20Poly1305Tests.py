import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import ChaCha20_Poly1305
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class ChaCha20Poly1305Tests(unittest.TestCase):
    key_256 = get_tag_random('key_256', 32)
    nonce_96 = get_tag_random('nonce_96', 12)
    data_128 = get_tag_random('data_128', 16)

    def test_loopback(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        pt = get_tag_random('plaintext', 16 * 100)
        ct = cipher.encrypt(pt)
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_nonce(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=b'H' * 8)
        self.assertEqual(len(cipher.nonce), 8)
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=b'H' * 12)
        self.assertEqual(len(cipher.nonce), 12)
        cipher = ChaCha20_Poly1305.new(key=self.key_256)
        nonce1 = cipher.nonce
        cipher = ChaCha20_Poly1305.new(key=self.key_256)
        nonce2 = cipher.nonce
        self.assertEqual(len(nonce1), 12)
        self.assertNotEqual(nonce1, nonce2)
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data_128)
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        self.assertEqual(ct, cipher.encrypt(self.data_128))

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, ChaCha20_Poly1305.new, key=self.key_256, nonce=u'test12345678')

    def test_nonce_length(self):
        self.assertRaises(ValueError, ChaCha20_Poly1305.new, key=self.key_256, nonce=b'0' * 7)
        self.assertRaises(ValueError, ChaCha20_Poly1305.new, key=self.key_256, nonce=b'')

    def test_block_size(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        self.assertFalse(hasattr(cipher, 'block_size'))

    def test_nonce_attribute(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)
        nonce1 = ChaCha20_Poly1305.new(key=self.key_256).nonce
        nonce2 = ChaCha20_Poly1305.new(key=self.key_256).nonce
        self.assertEqual(len(nonce1), 12)
        self.assertNotEqual(nonce1, nonce2)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, ChaCha20_Poly1305.new, key=self.key_256, param=9)

    def test_null_encryption_decryption(self):
        for func in ('encrypt', 'decrypt'):
            cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
            result = getattr(cipher, func)(b'')
            self.assertEqual(result, b'')

    def test_either_encrypt_or_decrypt(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        cipher.encrypt(b'')
        self.assertRaises(TypeError, cipher.decrypt, b'')
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        cipher.decrypt(b'')
        self.assertRaises(TypeError, cipher.encrypt, b'')

    def test_data_must_be_bytes(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_mac_len(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data_128)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Cryptodome.Util.strxor import strxor_c
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data_128)
        invalid_mac = strxor_c(mac, 1)
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct, invalid_mac)

    def test_hex_mac(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_message_chunks(self):
        auth_data = get_tag_random('authenticated data', 127)
        plaintext = get_tag_random('plaintext', 127)
        cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        cipher.update(auth_data)
        ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

        def break_up(data, chunk_length):
            return [data[i:i + chunk_length] for i in range(0, len(data), chunk_length)]
        for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
            cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            pt2 = b''
            for chunk in break_up(ciphertext, chunk_length):
                pt2 += cipher.decrypt(chunk)
            self.assertEqual(plaintext, pt2)
            cipher.verify(ref_mac)
        for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
            cipher = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            ct2 = b''
            for chunk in break_up(plaintext, chunk_length):
                ct2 += cipher.encrypt(chunk)
            self.assertEqual(ciphertext, ct2)
            self.assertEqual(cipher.digest(), ref_mac)

    def test_bytearray(self):
        key_ba = bytearray(self.key_256)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data_128)
        data_ba = bytearray(self.data_128)
        cipher1 = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        cipher1.update(self.data_128)
        ct = cipher1.encrypt(self.data_128)
        tag = cipher1.digest()
        cipher2 = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        key_ba[:3] = b'\xff\xff\xff'
        nonce_ba[:3] = b'\xff\xff\xff'
        cipher2.update(header_ba)
        header_ba[:3] = b'\xff\xff\xff'
        ct_test = cipher2.encrypt(data_ba)
        data_ba[:3] = b'\x99\x99\x99'
        tag_test = cipher2.digest()
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        key_ba = bytearray(self.key_256)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data_128)
        ct_ba = bytearray(ct)
        tag_ba = bytearray(tag)
        del data_ba
        cipher3 = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        key_ba[:3] = b'\xff\xff\xff'
        nonce_ba[:3] = b'\xff\xff\xff'
        cipher3.update(header_ba)
        header_ba[:3] = b'\xff\xff\xff'
        pt_test = cipher3.decrypt(ct_ba)
        ct_ba[:3] = b'\xff\xff\xff'
        cipher3.verify(tag_ba)
        self.assertEqual(pt_test, self.data_128)

    def test_memoryview(self):
        key_mv = memoryview(bytearray(self.key_256))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data_128))
        data_mv = memoryview(bytearray(self.data_128))
        cipher1 = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        cipher1.update(self.data_128)
        ct = cipher1.encrypt(self.data_128)
        tag = cipher1.digest()
        cipher2 = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        key_mv[:3] = b'\xff\xff\xff'
        nonce_mv[:3] = b'\xff\xff\xff'
        cipher2.update(header_mv)
        header_mv[:3] = b'\xff\xff\xff'
        ct_test = cipher2.encrypt(data_mv)
        data_mv[:3] = b'\x99\x99\x99'
        tag_test = cipher2.digest()
        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        key_mv = memoryview(bytearray(self.key_256))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data_128))
        ct_mv = memoryview(bytearray(ct))
        tag_mv = memoryview(bytearray(tag))
        del data_mv
        cipher3 = ChaCha20_Poly1305.new(key=self.key_256, nonce=self.nonce_96)
        key_mv[:3] = b'\xff\xff\xff'
        nonce_mv[:3] = b'\xff\xff\xff'
        cipher3.update(header_mv)
        header_mv[:3] = b'\xff\xff\xff'
        pt_test = cipher3.decrypt(ct_mv)
        ct_mv[:3] = b'\x99\x99\x99'
        cipher3.verify(tag_mv)
        self.assertEqual(pt_test, self.data_128)