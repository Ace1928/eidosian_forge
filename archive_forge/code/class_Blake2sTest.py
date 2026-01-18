import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2sTest(Blake2Test):
    BLAKE2 = BLAKE2s
    max_bits = 256
    max_bytes = 32
    digest_bits_oid = (128, 160, 224, 256)
    oid_variant = '2'