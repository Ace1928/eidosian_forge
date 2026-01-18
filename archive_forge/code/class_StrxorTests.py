import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
class StrxorTests(unittest.TestCase):

    def test1(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term2 = unhexlify(b'383d4ba020573314395b')
        result = unhexlify(b'c70ed123c59a7fcb6f12')
        self.assertEqual(strxor(term1, term2), result)
        self.assertEqual(strxor(term2, term1), result)

    def test2(self):
        es = b''
        self.assertEqual(strxor(es, es), es)

    def test3(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        all_zeros = b'\x00' * len(term1)
        self.assertEqual(strxor(term1, term1), all_zeros)

    def test_wrong_length(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term2 = unhexlify(b'ff339a83e5cd4cdf564990')
        self.assertRaises(ValueError, strxor, term1, term2)

    def test_bytearray(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term1_ba = bytearray(term1)
        term2 = unhexlify(b'383d4ba020573314395b')
        result = unhexlify(b'c70ed123c59a7fcb6f12')
        self.assertEqual(strxor(term1_ba, term2), result)

    def test_memoryview(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term1_mv = memoryview(term1)
        term2 = unhexlify(b'383d4ba020573314395b')
        result = unhexlify(b'c70ed123c59a7fcb6f12')
        self.assertEqual(strxor(term1_mv, term2), result)

    def test_output_bytearray(self):
        """Verify result can be stored in pre-allocated memory"""
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term2 = unhexlify(b'383d4ba020573314395b')
        original_term1 = term1[:]
        original_term2 = term2[:]
        expected_xor = unhexlify(b'c70ed123c59a7fcb6f12')
        output = bytearray(len(term1))
        result = strxor(term1, term2, output=output)
        self.assertEqual(result, None)
        self.assertEqual(output, expected_xor)
        self.assertEqual(term1, original_term1)
        self.assertEqual(term2, original_term2)

    def test_output_memoryview(self):
        """Verify result can be stored in pre-allocated memory"""
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term2 = unhexlify(b'383d4ba020573314395b')
        original_term1 = term1[:]
        original_term2 = term2[:]
        expected_xor = unhexlify(b'c70ed123c59a7fcb6f12')
        output = memoryview(bytearray(len(term1)))
        result = strxor(term1, term2, output=output)
        self.assertEqual(result, None)
        self.assertEqual(output, expected_xor)
        self.assertEqual(term1, original_term1)
        self.assertEqual(term2, original_term2)

    def test_output_overlapping_bytearray(self):
        """Verify result can be stored in overlapping memory"""
        term1 = bytearray(unhexlify(b'ff339a83e5cd4cdf5649'))
        term2 = unhexlify(b'383d4ba020573314395b')
        original_term2 = term2[:]
        expected_xor = unhexlify(b'c70ed123c59a7fcb6f12')
        result = strxor(term1, term2, output=term1)
        self.assertEqual(result, None)
        self.assertEqual(term1, expected_xor)
        self.assertEqual(term2, original_term2)

    def test_output_overlapping_memoryview(self):
        """Verify result can be stored in overlapping memory"""
        term1 = memoryview(bytearray(unhexlify(b'ff339a83e5cd4cdf5649')))
        term2 = unhexlify(b'383d4ba020573314395b')
        original_term2 = term2[:]
        expected_xor = unhexlify(b'c70ed123c59a7fcb6f12')
        result = strxor(term1, term2, output=term1)
        self.assertEqual(result, None)
        self.assertEqual(term1, expected_xor)
        self.assertEqual(term2, original_term2)

    def test_output_ro_bytes(self):
        """Verify result cannot be stored in read-only memory"""
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term2 = unhexlify(b'383d4ba020573314395b')
        self.assertRaises(TypeError, strxor, term1, term2, output=term1)

    def test_output_ro_memoryview(self):
        """Verify result cannot be stored in read-only memory"""
        term1 = memoryview(unhexlify(b'ff339a83e5cd4cdf5649'))
        term2 = unhexlify(b'383d4ba020573314395b')
        self.assertRaises(TypeError, strxor, term1, term2, output=term1)

    def test_output_incorrect_length(self):
        """Verify result cannot be stored in memory of incorrect length"""
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term2 = unhexlify(b'383d4ba020573314395b')
        output = bytearray(len(term1) - 1)
        self.assertRaises(ValueError, strxor, term1, term2, output=output)