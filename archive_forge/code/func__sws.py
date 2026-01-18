import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _sws(s):
    """Remove whitespace from a text or byte string"""
    if isinstance(s, str):
        return ''.join(s.split())
    else:
        return b('').join(s.split())