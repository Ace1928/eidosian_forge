from __future__ import print_function
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util.strxor import strxor
class GcmFSMTests(unittest.TestCase):
    key_128 = get_tag_random('key_128', 16)
    nonce_96 = get_tag_random('nonce_128', 12)
    data = get_tag_random('data', 128)

    def test_valid_init_encrypt_decrypt_digest_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_update_digest_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.verify(mac)

    def test_valid_full_path(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_digest(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_encrypt_or_decrypt(self):
        for method_name in ('encrypt', 'decrypt'):
            for auth_data in (None, b'333', self.data, self.data + b'3'):
                if auth_data is None:
                    assoc_len = None
                else:
                    assoc_len = len(auth_data)
                cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
                if auth_data is not None:
                    cipher.update(auth_data)
                method = getattr(cipher, method_name)
                method(self.data)
                method(self.data)
                method(self.data)
                method(self.data)

    def test_valid_multiple_digest_or_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        ct, mac = cipher.encrypt_and_digest(self.data)
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data, pt)

    def test_invalid_mixing_encrypt_decrypt(self):
        for method1_name, method2_name in (('encrypt', 'decrypt'), ('decrypt', 'encrypt')):
            for assoc_data_present in (True, False):
                cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
                if assoc_data_present:
                    cipher.update(self.data)
                getattr(cipher, method1_name)(self.data)
                self.assertRaises(TypeError, getattr(cipher, method2_name), self.data)

    def test_invalid_encrypt_or_update_after_digest(self):
        for method_name in ('encrypt', 'update'):
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.encrypt(self.data)
            cipher.digest()
            self.assertRaises(TypeError, getattr(cipher, method_name), self.data)
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.encrypt_and_digest(self.data)

    def test_invalid_decrypt_or_update_after_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()
        for method_name in ('decrypt', 'update'):
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name), self.data)
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.decrypt_and_verify(ct, mac)
            self.assertRaises(TypeError, getattr(cipher, method_name), self.data)