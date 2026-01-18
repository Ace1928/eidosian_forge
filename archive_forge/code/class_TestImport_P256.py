import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
class TestImport_P256(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestImport_P256, self).__init__(*args, **kwargs)
        self.ref_private, self.ref_public = create_ref_keys_p256()

    def test_import_public_der(self):
        key_file = load_file('ecc_p256_public.der')
        key = ECC._import_subjectPublicKeyInfo(key_file)
        self.assertEqual(self.ref_public, key)
        key = ECC._import_der(key_file, None)
        self.assertEqual(self.ref_public, key)
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_public, key)

    def test_import_sec1_uncompressed(self):
        key_file = load_file('ecc_p256_public.der')
        value = extract_bitstring_from_spki(key_file)
        key = ECC.import_key(key_file, curve_name='P256')
        self.assertEqual(self.ref_public, key)

    def test_import_sec1_compressed(self):
        key_file = load_file('ecc_p256_public_compressed.der')
        value = extract_bitstring_from_spki(key_file)
        key = ECC.import_key(key_file, curve_name='P256')
        self.assertEqual(self.ref_public, key)

    def test_import_rfc5915_der(self):
        key_file = load_file('ecc_p256_private.der')
        key = ECC._import_rfc5915_der(key_file, None)
        self.assertEqual(self.ref_private, key)
        key = ECC._import_der(key_file, None)
        self.assertEqual(self.ref_private, key)
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_private, key)

    def test_import_private_pkcs8_clear(self):
        key_file = load_file('ecc_p256_private_p8_clear.der')
        key = ECC._import_der(key_file, None)
        self.assertEqual(self.ref_private, key)
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_private, key)

    def test_import_private_pkcs8_in_pem_clear(self):
        key_file = load_file('ecc_p256_private_p8_clear.pem')
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_private, key)

    def test_import_private_pkcs8_encrypted_1(self):
        key_file = load_file('ecc_p256_private_p8.der')
        key = ECC._import_der(key_file, 'secret')
        self.assertEqual(self.ref_private, key)
        key = ECC.import_key(key_file, 'secret')
        self.assertEqual(self.ref_private, key)

    def test_import_private_pkcs8_encrypted_2(self):
        key_file = load_file('ecc_p256_private_p8.pem')
        key = ECC.import_key(key_file, 'secret')
        self.assertEqual(self.ref_private, key)

    def test_import_private_pkcs8_encrypted_3(self):
        key_file = load_file('ecc_p256_private_p8_2.der')
        key = ECC._import_der(key_file, 'secret')
        self.assertEqual(self.ref_private, key)
        key = ECC.import_key(key_file, 'secret')
        self.assertEqual(self.ref_private, key)

    def test_import_x509_der(self):
        key_file = load_file('ecc_p256_x509.der')
        key = ECC._import_der(key_file, None)
        self.assertEqual(self.ref_public, key)
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_public, key)

    def test_import_public_pem(self):
        key_file = load_file('ecc_p256_public.pem')
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_public, key)

    def test_import_private_pem(self):
        key_file = load_file('ecc_p256_private.pem')
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_private, key)

    def test_import_private_pem_with_ecparams(self):
        key_file = load_file('ecc_p256_private_ecparams.pem')
        key = ECC.import_key(key_file)

    def test_import_private_pem_encrypted(self):
        for algo in ('des3', 'aes128', 'aes192', 'aes256', 'aes256_gcm'):
            key_file = load_file('ecc_p256_private_enc_%s.pem' % algo)
            key = ECC.import_key(key_file, 'secret')
            self.assertEqual(self.ref_private, key)
            key = ECC.import_key(tostr(key_file), b'secret')
            self.assertEqual(self.ref_private, key)

    def test_import_x509_pem(self):
        key_file = load_file('ecc_p256_x509.pem')
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_public, key)

    def test_import_openssh_public(self):
        key_file = load_file('ecc_p256_public_openssh.txt')
        key = ECC._import_openssh_public(key_file)
        self.assertEqual(self.ref_public, key)
        key = ECC.import_key(key_file)
        self.assertEqual(self.ref_public, key)

    def test_import_openssh_private_clear(self):
        key_file = load_file('ecc_p256_private_openssh.pem')
        key_file_old = load_file('ecc_p256_private_openssh_old.pem')
        key = ECC.import_key(key_file)
        key_old = ECC.import_key(key_file_old)
        self.assertEqual(key, key_old)

    def test_import_openssh_private_password(self):
        key_file = load_file('ecc_p256_private_openssh_pwd.pem')
        key_file_old = load_file('ecc_p256_private_openssh_pwd_old.pem')
        key = ECC.import_key(key_file, b'password')
        key_old = ECC.import_key(key_file_old)
        self.assertEqual(key, key_old)