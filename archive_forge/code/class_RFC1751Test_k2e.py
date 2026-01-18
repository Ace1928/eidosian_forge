import binascii
import unittest
from Cryptodome.Util import RFC1751
from Cryptodome.Util.py3compat import *
class RFC1751Test_k2e(unittest.TestCase):

    def runTest(self):
        """Check converting keys to English"""
        for key, words in test_data:
            key = binascii.a2b_hex(b(key))
            self.assertEqual(RFC1751.key_to_english(key), words)