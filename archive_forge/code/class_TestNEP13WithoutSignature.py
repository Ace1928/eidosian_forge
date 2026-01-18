import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
class TestNEP13WithoutSignature(TestCase):

    def test_all(self):

        @vectorize(nopython=True)
        def new_ufunc(hundreds, tens, ones):
            return 100 * hundreds + 10 * tens + ones

        class NEP13Array:

            def __init__(self, array):
                self.array = array

            def __array__(self):
                return self.array

            def tolist(self):
                return self.array.tolist()

            def __array_ufunc__(self, ufunc, method, *args, **kwargs):
                if method != '__call__':
                    return NotImplemented
                return NEP13Array(ufunc(*[np.asarray(x) for x in args], **kwargs))
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        c = np.array([7, 8, 9], dtype=np.int64)
        all_np = new_ufunc(a, b, c)
        self.assertIsInstance(all_np, np.ndarray)
        self.assertEqual(all_np.tolist(), [147, 258, 369])
        nep13_1 = new_ufunc(NEP13Array(a), b, c)
        self.assertIsInstance(nep13_1, NEP13Array)
        self.assertEqual(nep13_1.tolist(), [147, 258, 369])
        nep13_2 = new_ufunc(a, NEP13Array(b), c)
        self.assertIsInstance(nep13_2, NEP13Array)
        self.assertEqual(nep13_2.tolist(), [147, 258, 369])
        nep13_3 = new_ufunc(a, b, NEP13Array(c))
        self.assertIsInstance(nep13_3, NEP13Array)
        self.assertEqual(nep13_3.tolist(), [147, 258, 369])
        a = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        b = np.array([4.4, 5.5, 6.6], dtype=np.float64)
        c = np.array([7.7, 8.8, 9.9], dtype=np.float64)
        all_np = new_ufunc(a, b, c)
        self.assertIsInstance(all_np, np.ndarray)
        self.assertEqual(all_np.tolist(), [161.7, 283.8, 405.9])
        nep13_1 = new_ufunc(NEP13Array(a), b, c)
        self.assertIsInstance(nep13_1, NEP13Array)
        self.assertEqual(nep13_1.tolist(), [161.7, 283.8, 405.9])
        nep13_2 = new_ufunc(a, NEP13Array(b), c)
        self.assertIsInstance(nep13_2, NEP13Array)
        self.assertEqual(nep13_2.tolist(), [161.7, 283.8, 405.9])
        nep13_3 = new_ufunc(a, b, NEP13Array(c))
        self.assertIsInstance(nep13_3, NEP13Array)
        self.assertEqual(nep13_3.tolist(), [161.7, 283.8, 405.9])