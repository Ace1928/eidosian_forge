import re
import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import bord
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Protocol.DH import key_agreement
class FIPS_ECDH_Tests_KAT(unittest.TestCase):
    pass