import os
import re
import errno
import warnings
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import a2b_hex, list_test_cases
from Cryptodome.IO import PEM
from Cryptodome.Util.py3compat import b, tostr, FileNotFoundError
from Cryptodome.Util.number import inverse, bytes_to_long
from Cryptodome.Util import asn1
class TestImport_2048(unittest.TestCase):

    def test_import_openssh_public(self):
        key_file_ref = load_file('rsa2048_private.pem')
        key_file = load_file('rsa2048_public_openssh.txt')
        if None in (key_file_ref, key_file):
            return
        key_ref = RSA.import_key(key_file_ref).public_key()
        key = RSA.import_key(key_file)
        self.assertEqual(key_ref, key)

    def test_import_openssh_private_clear(self):
        key_file = load_file('rsa2048_private_openssh.pem')
        key_file_old = load_file('rsa2048_private_openssh_old.pem')
        if None in (key_file_old, key_file):
            return
        key = RSA.import_key(key_file)
        key_old = RSA.import_key(key_file_old)
        self.assertEqual(key, key_old)

    def test_import_openssh_private_password(self):
        key_file = load_file('rsa2048_private_openssh_pwd.pem')
        key_file_old = load_file('rsa2048_private_openssh_pwd_old.pem')
        if None in (key_file_old, key_file):
            return
        key = RSA.import_key(key_file, b'password')
        key_old = RSA.import_key(key_file_old)
        self.assertEqual(key, key_old)

    def test_import_pkcs8_private(self):
        key_file_ref = load_file('rsa2048_private.pem')
        key_file = load_file('rsa2048_private_p8.der')
        if None in (key_file_ref, key_file):
            return
        key_ref = RSA.import_key(key_file_ref)
        key = RSA.import_key(key_file, b'secret')
        self.assertEqual(key_ref, key)