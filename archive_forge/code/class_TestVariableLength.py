from __future__ import print_function
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util.strxor import strxor
class TestVariableLength(unittest.TestCase):

    def __init__(self, **extra_params):
        unittest.TestCase.__init__(self)
        self._extra_params = extra_params

    def runTest(self):
        key = b'0' * 16
        h = SHA256.new()
        for length in range(160):
            nonce = '{0:04d}'.format(length).encode('utf-8')
            data = bchr(length) * length
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, **self._extra_params)
            ct, tag = cipher.encrypt_and_digest(data)
            h.update(ct)
            h.update(tag)
        self.assertEqual(h.hexdigest(), '7b7eb1ffbe67a2e53a912067c0ec8e62ebc7ce4d83490ea7426941349811bdf4')