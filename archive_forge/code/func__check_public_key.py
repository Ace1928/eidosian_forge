import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _check_public_key(self, dsaObj):
    k = bytes_to_long(a2b_hex(self.k))
    m_hash = bytes_to_long(a2b_hex(self.m_hash))
    self.assertEqual(0, dsaObj.has_private())
    self.assertEqual(1, dsaObj.can_sign())
    self.assertEqual(0, dsaObj.can_encrypt())
    self.assertEqual(0, hasattr(dsaObj, 'x'))
    self.assertEqual(1, dsaObj.p > dsaObj.q)
    self.assertEqual(160, size(dsaObj.q))
    self.assertEqual(0, (dsaObj.p - 1) % dsaObj.q)
    self.assertRaises(TypeError, dsaObj._sign, m_hash, k)
    self.assertEqual(dsaObj.public_key() == dsaObj.public_key(), True)
    self.assertEqual(dsaObj.public_key() != dsaObj.public_key(), False)
    self.assertEqual(dsaObj.public_key(), dsaObj.publickey())