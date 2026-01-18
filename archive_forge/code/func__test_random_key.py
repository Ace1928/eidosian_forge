import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
from Cryptodome import Random
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.py3compat import *
def _test_random_key(self, bits):
    elgObj = ElGamal.generate(bits, Random.new().read)
    self._check_private_key(elgObj)
    self._exercise_primitive(elgObj)
    pub = elgObj.publickey()
    self._check_public_key(pub)
    self._exercise_public_primitive(elgObj)