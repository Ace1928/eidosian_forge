from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
class TestUnicodeNames(TestCase):

    def make_testcase(self, src, fname):
        glb = {}
        exec(src, glb)
        fn = glb[fname]
        return fn

    def test_unicode_name1(self):
        fn = self.make_testcase(unicode_name1, 'unicode_name1')
        cfn = njit(fn)
        self.assertEqual(cfn(1, 2), 3)

    def test_unicode_name2(self):
        fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
        cfn = njit(fn)
        self.assertEqual(cfn(1, 2), 3)

    def test_cfunc(self):
        fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
        cfn = cfunc('int32(int32, int32)')(fn)
        self.assertEqual(cfn.ctypes(1, 2), 3)