import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _check_decryption(self, rsaObj):
    plaintext = bytes_to_long(a2b_hex(self.plaintext))
    ciphertext = bytes_to_long(a2b_hex(self.ciphertext))
    new_plaintext = rsaObj._decrypt(ciphertext)
    self.assertEqual(plaintext, new_plaintext)