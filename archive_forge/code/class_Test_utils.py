import unittest
from zope.interface.tests import OptimizationTestMixin
class Test_utils(unittest.TestCase):

    def test__convert_None_to_Interface_w_None(self):
        from zope.interface.adapter import _convert_None_to_Interface
        from zope.interface.interface import Interface
        self.assertIs(_convert_None_to_Interface(None), Interface)

    def test__convert_None_to_Interface_w_other(self):
        from zope.interface.adapter import _convert_None_to_Interface
        other = object()
        self.assertIs(_convert_None_to_Interface(other), other)

    def test__normalize_name_str(self):
        from zope.interface.adapter import _normalize_name
        STR = b'str'
        UNICODE = 'str'
        norm = _normalize_name(STR)
        self.assertEqual(norm, UNICODE)
        self.assertIsInstance(norm, type(UNICODE))

    def test__normalize_name_unicode(self):
        from zope.interface.adapter import _normalize_name
        USTR = 'ustr'
        self.assertEqual(_normalize_name(USTR), USTR)

    def test__normalize_name_other(self):
        from zope.interface.adapter import _normalize_name
        for other in (1, 1.0, (), [], {}, object()):
            self.assertRaises(TypeError, _normalize_name, other)