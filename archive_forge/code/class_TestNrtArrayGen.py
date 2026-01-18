import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
class TestNrtArrayGen(MemoryLeakMixin, TestCase):

    def test_nrt_gen0(self):
        pygen = nrt_gen0
        cgen = jit(nopython=True)(pygen)
        py_ary = np.arange(10)
        c_ary = py_ary.copy()
        py_res = list(pygen(py_ary))
        c_res = list(cgen(c_ary))
        np.testing.assert_equal(py_ary, c_ary)
        self.assertEqual(py_res, c_res)
        self.assertRefCountEqual(py_ary, c_ary)

    def test_nrt_gen1(self):
        pygen = nrt_gen1
        cgen = jit(nopython=True)(pygen)
        py_ary1 = np.arange(10)
        py_ary2 = py_ary1 + 100
        c_ary1 = py_ary1.copy()
        c_ary2 = py_ary2.copy()
        py_res = list(pygen(py_ary1, py_ary2))
        c_res = list(cgen(c_ary1, c_ary2))
        np.testing.assert_equal(py_ary1, c_ary1)
        np.testing.assert_equal(py_ary2, c_ary2)
        self.assertEqual(py_res, c_res)
        self.assertRefCountEqual(py_ary1, c_ary1)
        self.assertRefCountEqual(py_ary2, c_ary2)

    def test_combine_gen0_gen1(self):
        """
        Issue #1163 is observed when two generator with NRT object arguments
        is ran in sequence.  The first one does a invalid free and corrupts
        the NRT memory subsystem.  The second generator is likely to segfault
        due to corrupted NRT data structure (an invalid MemInfo).
        """
        self.test_nrt_gen0()
        self.test_nrt_gen1()

    def test_nrt_gen0_stop_iteration(self):
        """
        Test cleanup on StopIteration
        """
        pygen = nrt_gen0
        cgen = jit(nopython=True)(pygen)
        py_ary = np.arange(1)
        c_ary = py_ary.copy()
        py_iter = pygen(py_ary)
        c_iter = cgen(c_ary)
        py_res = next(py_iter)
        c_res = next(c_iter)
        with self.assertRaises(StopIteration):
            py_res = next(py_iter)
        with self.assertRaises(StopIteration):
            c_res = next(c_iter)
        del py_iter
        del c_iter
        np.testing.assert_equal(py_ary, c_ary)
        self.assertEqual(py_res, c_res)
        self.assertRefCountEqual(py_ary, c_ary)

    def test_nrt_gen0_no_iter(self):
        """
        Test cleanup for a initialized but never iterated (never call next())
        generator.
        """
        pygen = nrt_gen0
        cgen = jit(nopython=True)(pygen)
        py_ary = np.arange(1)
        c_ary = py_ary.copy()
        py_iter = pygen(py_ary)
        c_iter = cgen(c_ary)
        del py_iter
        del c_iter
        np.testing.assert_equal(py_ary, c_ary)
        self.assertRefCountEqual(py_ary, c_ary)