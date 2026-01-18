import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
class TupleHash128Test(TupleHashTest):
    TupleHash = TupleHash128
    minimum_bytes = 8
    default_bytes = 64
    minimum_bits = 64
    default_bits = 512