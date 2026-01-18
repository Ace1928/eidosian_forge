import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Cipher import DES3
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.Util.py3compat import bchr, tostr
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
class CheckParity(unittest.TestCase):

    def test_parity_option2(self):
        before_2k = unhexlify('CABF326FA56734324FFCCABCDEFACABF')
        after_2k = DES3.adjust_key_parity(before_2k)
        self.assertEqual(after_2k, unhexlify('CBBF326EA46734324FFDCBBCDFFBCBBF'))

    def test_parity_option3(self):
        before_3k = unhexlify('AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCC')
        after_3k = DES3.adjust_key_parity(before_3k)
        self.assertEqual(after_3k, unhexlify('ABABABABABABABABBABABABABABABABACDCDCDCDCDCDCDCD'))

    def test_degradation(self):
        sub_key1 = bchr(1) * 8
        sub_key2 = bchr(255) * 8
        self.assertRaises(ValueError, DES3.adjust_key_parity, sub_key1 * 2 + sub_key2)
        self.assertRaises(ValueError, DES3.adjust_key_parity, sub_key1 + sub_key2 * 2)
        self.assertRaises(ValueError, DES3.adjust_key_parity, sub_key1 * 3)
        self.assertRaises(ValueError, DES3.adjust_key_parity, sub_key1 + strxor_c(sub_key1, 1) + sub_key2)