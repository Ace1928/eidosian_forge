import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
class TestBoundsCheckError(TestCase):

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
    def test_basic_array_boundscheck(self):
        self.assertIsNone(config.BOUNDSCHECK)
        a = np.arange(5)
        with self.assertRaises(IndexError):
            basic_array_access(a)
        at = typeof(a)
        boundscheck = njit((at,), boundscheck=True)(basic_array_access)
        with self.assertRaises(IndexError):
            boundscheck(a)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
    def test_slice_array_boundscheck(self):
        self.assertIsNone(config.BOUNDSCHECK)
        a = np.ones((5, 5))
        b = np.ones((5, 20))
        with self.assertRaises(IndexError):
            slice_array_access(a)
        slice_array_access(b)
        at = typeof(a)
        rt = float64[:]
        boundscheck = njit(rt(at), boundscheck=True)(slice_array_access)
        with self.assertRaises(IndexError):
            boundscheck(a)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
    def test_fancy_indexing_boundscheck(self):
        self.assertIsNone(config.BOUNDSCHECK)
        a = np.arange(3)
        b = np.arange(4)
        with self.assertRaises(IndexError):
            fancy_array_access(a)
        fancy_array_access(b)
        at = typeof(a)
        rt = at.dtype[:]
        boundscheck = njit(rt(at), boundscheck=True)(fancy_array_access)
        with self.assertRaises(IndexError):
            boundscheck(a)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
    def test_fancy_indexing_with_modification_boundscheck(self):
        self.assertIsNone(config.BOUNDSCHECK)
        a = np.arange(3)
        b = np.arange(4)
        with self.assertRaises(IndexError):
            fancy_array_modify(a)
        fancy_array_modify(b)
        at = typeof(a)
        rt = at.dtype[:]
        boundscheck = njit(rt(at), boundscheck=True)(fancy_array_modify)
        with self.assertRaises(IndexError):
            boundscheck(a)