import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _test_verification(self, dsaObj):
    m_hash = bytes_to_long(a2b_hex(self.m_hash))
    r = bytes_to_long(a2b_hex(self.r))
    s = bytes_to_long(a2b_hex(self.s))
    self.assertTrue(dsaObj._verify(m_hash, (r, s)))
    self.assertFalse(dsaObj._verify(m_hash + 1, (r, s)))