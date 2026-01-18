import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _exercise_public_primitive(self, rsaObj):
    plaintext = a2b_hex(self.plaintext)
    new_ciphertext2 = rsaObj._encrypt(bytes_to_long(plaintext))