from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
class TestUnicodeUtils(TestCase):

    def test_normalize_ir_text(self):
        out = cgutils.normalize_ir_text('abc')
        self.assertIsInstance(out, str)
        out.encode('latin1')

    def test_normalize_ir_text_unicode(self):
        out = cgutils.normalize_ir_text(unicode_name2)
        self.assertIsInstance(out, str)
        out.encode('latin1')