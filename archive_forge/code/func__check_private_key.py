import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _check_private_key(self, dsaObj):
    self.assertEqual(1, dsaObj.has_private())
    self.assertEqual(1, dsaObj.can_sign())
    self.assertEqual(0, dsaObj.can_encrypt())
    self.assertEqual(1, dsaObj.p > dsaObj.q)
    self.assertEqual(160, size(dsaObj.q))
    self.assertEqual(0, (dsaObj.p - 1) % dsaObj.q)
    self.assertEqual(dsaObj.y, pow(dsaObj.g, dsaObj.x, dsaObj.p))
    self.assertEqual(1, 0 < dsaObj.x < dsaObj.q)