import unittest
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import cSHAKE128, cSHAKE256, SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, tobytes
class cSHAKE128Test(cSHAKETest):
    cshake = cSHAKE128
    shake = SHAKE128