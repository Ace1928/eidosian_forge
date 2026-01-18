import unittest
from ctypes import *
from ctypes.test import need_symbol
@need_symbol('c_wchar')
class WStringTestCase(unittest.TestCase):

    def test_wchar(self):
        c_wchar('x')
        repr(byref(c_wchar('x')))
        c_wchar('x')

    @unittest.skip('test disabled')
    def test_basic_wstrings(self):
        cs = c_wstring('abcdef')
        self.assertEqual(sizeof(cs), 14)
        self.assertEqual(cs.value, 'abcdef')
        self.assertEqual(c_wstring('abc\x00def').value, 'abc')
        self.assertEqual(c_wstring('abc\x00def').value, 'abc')
        self.assertEqual(cs.raw, 'abcdef\x00')
        self.assertEqual(c_wstring('abc\x00def').raw, 'abc\x00def\x00')
        cs.value = 'ab'
        self.assertEqual(cs.value, 'ab')
        self.assertEqual(cs.raw, 'ab\x00\x00\x00\x00\x00')
        self.assertRaises(TypeError, c_wstring, '123')
        self.assertRaises(ValueError, c_wstring, 0)

    @unittest.skip('test disabled')
    def test_toolong(self):
        cs = c_wstring('abcdef')
        self.assertRaises(ValueError, setattr, cs, 'value', '123456789012345')
        self.assertRaises(ValueError, setattr, cs, 'value', '1234567')