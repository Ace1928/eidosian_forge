import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _exercise_primitive(self, rsaObj):
    ciphertext = bytes_to_long(a2b_hex(self.ciphertext))
    plaintext = rsaObj._decrypt(ciphertext)
    new_ciphertext2 = rsaObj._encrypt(plaintext)
    self.assertEqual(ciphertext, new_ciphertext2)