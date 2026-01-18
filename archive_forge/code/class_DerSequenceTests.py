import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerSequenceTests(unittest.TestCase):

    def testInit1(self):
        der = DerSequence([1, DerInteger(2), b('0\x00')])
        self.assertEqual(der.encode(), b('0\x08\x02\x01\x01\x02\x01\x020\x00'))

    def testEncode1(self):
        der = DerSequence()
        self.assertEqual(der.encode(), b('0\x00'))
        self.assertFalse(der.hasOnlyInts())
        der.append(0)
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x00'))
        self.assertEqual(der.hasInts(), 1)
        self.assertEqual(der.hasInts(False), 1)
        self.assertTrue(der.hasOnlyInts())
        self.assertTrue(der.hasOnlyInts(False))
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x00'))

    def testEncode2(self):
        der = DerSequence()
        der.append(0)
        der[0] = 1
        self.assertEqual(len(der), 1)
        self.assertEqual(der[0], 1)
        self.assertEqual(der[-1], 1)
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x01'))
        der[:] = [1]
        self.assertEqual(len(der), 1)
        self.assertEqual(der[0], 1)
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x01'))

    def testEncode3(self):
        der = DerSequence()
        der.append(384)
        self.assertEqual(der.encode(), b('0\x04\x02\x02\x01\x80'))

    def testEncode4(self):
        der = DerSequence()
        der.append(2 ** 2048)
        self.assertEqual(der.encode(), b('0\x82\x01\x05') + b('\x02\x82\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00'))

    def testEncode5(self):
        der = DerSequence()
        der += 1
        der += b('0\x00')
        self.assertEqual(der.encode(), b('0\x05\x02\x01\x010\x00'))

    def testEncode6(self):
        der = DerSequence()
        der.append(384)
        der.append(255)
        self.assertEqual(der.encode(), b('0\x08\x02\x02\x01\x80\x02\x02\x00ÿ'))
        self.assertTrue(der.hasOnlyInts())
        self.assertTrue(der.hasOnlyInts(False))
        der = DerSequence()
        der.append(2)
        der.append(-2)
        self.assertEqual(der.encode(), b('0\x06\x02\x01\x02\x02\x01þ'))
        self.assertEqual(der.hasInts(), 1)
        self.assertEqual(der.hasInts(False), 2)
        self.assertFalse(der.hasOnlyInts())
        self.assertTrue(der.hasOnlyInts(False))
        der.append(1)
        der[1:] = [9, 8]
        self.assertEqual(len(der), 3)
        self.assertEqual(der[1:], [9, 8])
        self.assertEqual(der[1:-1], [9])
        self.assertEqual(der.encode(), b('0\t\x02\x01\x02\x02\x01\t\x02\x01\x08'))

    def testEncode7(self):
        der = DerSequence()
        der.append(384)
        der.append(b('0\x03\x02\x01\x05'))
        self.assertEqual(der.encode(), b('0\t\x02\x02\x01\x800\x03\x02\x01\x05'))
        self.assertFalse(der.hasOnlyInts())

    def testEncode8(self):
        der = DerSequence()
        der.append(384)
        der.append(DerSequence([5]))
        self.assertEqual(der.encode(), b('0\t\x02\x02\x01\x800\x03\x02\x01\x05'))
        self.assertFalse(der.hasOnlyInts())

    def testDecode1(self):
        der = DerSequence()
        der.decode(b('0\x00'))
        self.assertEqual(len(der), 0)
        der.decode(b('0\x03\x02\x01\x00'))
        self.assertEqual(len(der), 1)
        self.assertEqual(der[0], 0)
        der.decode(b('0\x03\x02\x01\x00'))
        self.assertEqual(len(der), 1)
        self.assertEqual(der[0], 0)

    def testDecode2(self):
        der = DerSequence()
        der.decode(b('0\x03\x02\x01\x7f'))
        self.assertEqual(len(der), 1)
        self.assertEqual(der[0], 127)

    def testDecode4(self):
        der = DerSequence()
        der.decode(b('0\x82\x01\x05') + b('\x02\x82\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00') + b('\x00\x00\x00\x00\x00\x00\x00\x00\x00'))
        self.assertEqual(len(der), 1)
        self.assertEqual(der[0], 2 ** 2048)

    def testDecode6(self):
        der = DerSequence()
        der.decode(b('0\x08\x02\x02\x01\x80\x02\x02\x00ÿ'))
        self.assertEqual(len(der), 2)
        self.assertEqual(der[0], 384)
        self.assertEqual(der[1], 255)

    def testDecode7(self):
        der = DerSequence()
        der.decode(b('0\n\x02\x02\x01\x80$\x02¶c\x12\x00'))
        self.assertEqual(len(der), 3)
        self.assertEqual(der[0], 384)
        self.assertEqual(der[1], b('$\x02¶c'))
        self.assertEqual(der[2], b('\x12\x00'))

    def testDecode8(self):
        der = DerSequence()
        der.decode(b('0\x06$\x02¶c\x12\x00'))
        self.assertEqual(len(der), 2)
        self.assertEqual(der[0], b('$\x02¶c'))
        self.assertEqual(der[1], b('\x12\x00'))
        self.assertEqual(der.hasInts(), 0)
        self.assertEqual(der.hasInts(False), 0)
        self.assertFalse(der.hasOnlyInts())
        self.assertFalse(der.hasOnlyInts(False))

    def testDecode9(self):
        der = DerSequence()
        self.assertEqual(der, der.decode(b('0\x06$\x02¶c\x12\x00')))

    def testErrDecode1(self):
        der = DerSequence()
        self.assertRaises(ValueError, der.decode, b(''))
        self.assertRaises(ValueError, der.decode, b('\x00'))
        self.assertRaises(ValueError, der.decode, b('0'))

    def testErrDecode2(self):
        der = DerSequence()
        self.assertRaises(ValueError, der.decode, b('0\x00\x00'))

    def testErrDecode3(self):
        der = DerSequence()
        self.assertRaises(ValueError, der.decode, b('0\x04\x02\x01\x01\x00'))
        self.assertRaises(ValueError, der.decode, b('0\x81\x03\x02\x01\x01'))
        self.assertRaises(ValueError, der.decode, b('0\x04\x02\x81\x01\x01'))

    def test_expected_nr_elements(self):
        der_bin = DerSequence([1, 2, 3]).encode()
        DerSequence().decode(der_bin, nr_elements=3)
        DerSequence().decode(der_bin, nr_elements=(2, 3))
        self.assertRaises(ValueError, DerSequence().decode, der_bin, nr_elements=1)
        self.assertRaises(ValueError, DerSequence().decode, der_bin, nr_elements=(4, 5))

    def test_expected_only_integers(self):
        der_bin1 = DerSequence([1, 2, 3]).encode()
        der_bin2 = DerSequence([1, 2, DerSequence([3, 4])]).encode()
        DerSequence().decode(der_bin1, only_ints_expected=True)
        DerSequence().decode(der_bin1, only_ints_expected=False)
        DerSequence().decode(der_bin2, only_ints_expected=False)
        self.assertRaises(ValueError, DerSequence().decode, der_bin2, only_ints_expected=True)