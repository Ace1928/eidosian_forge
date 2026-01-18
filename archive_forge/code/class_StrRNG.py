import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
class StrRNG:

    def __init__(self, randomness):
        length = len(randomness)
        self._idx = 0
        self._randomness = long_to_bytes(bytes_to_long(randomness) - 1, length)

    def __call__(self, n):
        out = self._randomness[self._idx:self._idx + n]
        self._idx += n
        return out