import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2bTest(Blake2Test):
    BLAKE2 = BLAKE2b
    max_bits = 512
    max_bytes = 64
    digest_bits_oid = (160, 256, 384, 512)
    oid_variant = '1'