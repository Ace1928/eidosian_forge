import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
class EaxFSMTests(unittest.TestCase):
    key_128 = get_tag_random('key_128', 16)
    nonce_96 = get_tag_random('nonce_128', 12)
    data_128 = get_tag_random('data_128', 16)

    def test_valid_init_encrypt_decrypt_digest_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data_128)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_update_digest_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        cipher.verify(mac)

    def test_valid_full_path(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        ct = cipher.encrypt(self.data_128)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_digest(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        mac = cipher.digest()
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_encrypt_or_decrypt(self):
        for method_name in ('encrypt', 'decrypt'):
            for auth_data in (None, b'333', self.data_128, self.data_128 + b'3'):
                if auth_data is None:
                    assoc_len = None
                else:
                    assoc_len = len(auth_data)
                cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
                if auth_data is not None:
                    cipher.update(auth_data)
                method = getattr(cipher, method_name)
                method(self.data_128)
                method(self.data_128)
                method(self.data_128)
                method(self.data_128)

    def test_valid_multiple_digest_or_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        ct, mac = cipher.encrypt_and_digest(self.data_128)
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        cipher.update(self.data_128)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data_128, pt)

    def test_invalid_mixing_encrypt_decrypt(self):
        for method1_name, method2_name in (('encrypt', 'decrypt'), ('decrypt', 'encrypt')):
            for assoc_data_present in (True, False):
                cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
                if assoc_data_present:
                    cipher.update(self.data_128)
                getattr(cipher, method1_name)(self.data_128)
                self.assertRaises(TypeError, getattr(cipher, method2_name), self.data_128)

    def test_invalid_encrypt_or_update_after_digest(self):
        for method_name in ('encrypt', 'update'):
            cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
            cipher.encrypt(self.data_128)
            cipher.digest()
            self.assertRaises(TypeError, getattr(cipher, method_name), self.data_128)
            cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
            cipher.encrypt_and_digest(self.data_128)

    def test_invalid_decrypt_or_update_after_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data_128)
        mac = cipher.digest()
        for method_name in ('decrypt', 'update'):
            cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name), self.data_128)
            cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
            cipher.decrypt_and_verify(ct, mac)
            self.assertRaises(TypeError, getattr(cipher, method_name), self.data_128)