import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
class CtrTests(unittest.TestCase):
    key_128 = get_tag_random('key_128', 16)
    key_192 = get_tag_random('key_192', 24)
    nonce_32 = get_tag_random('nonce_32', 4)
    nonce_64 = get_tag_random('nonce_64', 8)
    ctr_64 = Counter.new(32, prefix=nonce_32)
    ctr_128 = Counter.new(64, prefix=nonce_64)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        pt = get_tag_random('plaintext', 16 * 100)
        ct = cipher.encrypt(pt)
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_loopback_64(self):
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        pt = get_tag_random('plaintext', 8 * 100)
        ct = cipher.encrypt(pt)
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_invalid_counter_parameter(self):
        self.assertRaises(TypeError, DES3.new, self.key_192, AES.MODE_CTR)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, self.ctr_128)

    def test_nonce_attribute(self):
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(cipher.nonce, self.nonce_32)
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(cipher.nonce, self.nonce_64)
        counter = Counter.new(64, prefix=self.nonce_32, suffix=self.nonce_32)
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertFalse(hasattr(cipher, 'nonce'))

    def test_nonce_parameter(self):
        cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertEqual(cipher1.nonce, self.nonce_64)
        counter = Counter.new(64, prefix=self.nonce_64, initial_value=0)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        pt = get_tag_random('plaintext', 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
        nonce1 = AES.new(self.key_128, AES.MODE_CTR).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_CTR).nonce
        self.assertNotEqual(nonce1, nonce2)
        self.assertEqual(len(nonce1), 8)
        cipher = AES.new(self.key_128, AES.MODE_CTR, nonce=b'')
        self.assertEqual(b'', cipher.nonce)
        cipher.encrypt(b'0' * 300)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, counter=self.ctr_128, nonce=self.nonce_64)

    def test_initial_value_parameter(self):
        cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=65535)
        counter = Counter.new(64, prefix=self.nonce_64, initial_value=65535)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        pt = get_tag_random('plaintext', 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
        cipher1 = AES.new(self.key_128, AES.MODE_CTR, initial_value=65535)
        counter = Counter.new(64, prefix=cipher1.nonce, initial_value=65535)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        pt = get_tag_random('plaintext', 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, counter=self.ctr_128, initial_value=0)

    def test_initial_value_bytes_parameter(self):
        cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=b'\x00' * 6 + b'\xff\xff')
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=65535)
        pt = get_tag_random('plaintext', 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, initial_value=b'5' * 17)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=b'5' * 9)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, initial_value=b'5' * 15)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=b'5' * 7)

    def test_iv_with_matching_length(self):
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, counter=Counter.new(120))
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR, counter=Counter.new(136))

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_block_size_64(self):
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(cipher.block_size, DES3.block_size)

    def test_unaligned_data_128(self):
        plaintexts = [b'7777777'] * 100
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [b'7777777'] * 100
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(b''.join(ciphertexts), cipher.encrypt(b''.join(plaintexts)))

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, 7, counter=self.ctr_128)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, counter=self.ctr_128, unknown=7)
        AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128, use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in ('encrypt', 'decrypt'):
            cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
            result = getattr(cipher, func)(b'')
            self.assertEqual(result, b'')

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        cipher.encrypt(b'')
        self.assertRaises(TypeError, cipher.decrypt, b'')
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        cipher.decrypt(b'')
        self.assertRaises(TypeError, cipher.encrypt, b'')

    def test_wrap_around(self):
        counter = Counter.new(8, prefix=bchr(9) * 15)
        max_bytes = 4096
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        cipher.encrypt(b'9' * max_bytes)
        self.assertRaises(OverflowError, cipher.encrypt, b'9')
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertRaises(OverflowError, cipher.encrypt, b'9' * (max_bytes + 1))
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        cipher.decrypt(b'9' * max_bytes)
        self.assertRaises(OverflowError, cipher.decrypt, b'9')
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertRaises(OverflowError, cipher.decrypt, b'9' * (max_bytes + 1))

    def test_bytearray(self):
        data = b'1' * 16
        iv = b'\x00' * 6 + b'\xff\xff'
        cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=iv)
        ref1 = cipher1.encrypt(data)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, nonce=bytearray(self.nonce_64), initial_value=bytearray(iv))
        ref2 = cipher2.encrypt(bytearray(data))
        self.assertEqual(ref1, ref2)
        self.assertEqual(cipher1.nonce, cipher2.nonce)
        cipher3 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64, initial_value=iv)
        ref3 = cipher3.decrypt(data)
        cipher4 = AES.new(self.key_128, AES.MODE_CTR, nonce=bytearray(self.nonce_64), initial_value=bytearray(iv))
        ref4 = cipher4.decrypt(bytearray(data))
        self.assertEqual(ref3, ref4)

    def test_very_long_data(self):
        cipher = AES.new(b'A' * 32, AES.MODE_CTR, nonce=b'')
        ct = cipher.encrypt(b'B' * 1000000)
        digest = SHA256.new(ct).hexdigest()
        self.assertEqual(digest, '96204fc470476561a3a8f3b6fe6d24be85c87510b638142d1d0fb90989f8a6a6')

    def test_output_param(self):
        pt = b'5' * 128
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        ct = cipher.encrypt(pt)
        output = bytearray(128)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):
        pt = b'5' * 128
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        ct = cipher.encrypt(pt)
        output = memoryview(bytearray(128))
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128
        pt = b'5' * LEN_PT
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        ct = cipher.encrypt(pt)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * LEN_PT)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * LEN_PT)
        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        cipher = AES.new(b'4' * 16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)