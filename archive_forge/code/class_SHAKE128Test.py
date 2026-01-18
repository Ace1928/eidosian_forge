import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, bord, tobytes
class SHAKE128Test(SHAKETest):
    shake = SHAKE128