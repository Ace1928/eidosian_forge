import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
class TestArrayAnalysisParallelRequired(TestCase):
    """This is to just split out tests that need the parallel backend and
    therefore serialised execution.
    """
    _numba_parallel_test_ = False

    @skip_unsupported
    def test_misc(self):

        @njit
        def swap(x, y):
            return (y, x)

        def test_bug2537(m):
            a = np.ones(m)
            b = np.ones(m)
            for i in range(m):
                a[i], b[i] = swap(a[i], b[i])
        try:
            njit(test_bug2537, parallel=True)(10)
        except IndexError:
            self.fail('test_bug2537 raised IndexError!')

    @skip_unsupported
    def test_global_namedtuple(self):
        Row = namedtuple('Row', ['A'])
        row = Row(3)

        def test_impl():
            rr = row
            res = rr.A
            if res == 2:
                res = 3
            return res
        self.assertEqual(njit(test_impl, parallel=True)(), test_impl())

    @skip_unsupported
    def test_array_T_issue_3700(self):

        def test_impl(t_obj, X):
            for i in prange(t_obj.T):
                X[i] = i
            return X.sum()
        n = 5
        t_obj = ExampleClass3700(n)
        X1 = np.zeros(t_obj.T)
        X2 = np.zeros(t_obj.T)
        self.assertEqual(njit(test_impl, parallel=True)(t_obj, X1), test_impl(t_obj, X2))

    @skip_unsupported
    def test_slice_shape_issue_3380(self):

        def test_impl1():
            a = slice(None, None)
            return True
        self.assertEqual(njit(test_impl1, parallel=True)(), test_impl1())

        def test_impl2(A, a):
            b = a
            return A[b]
        A = np.arange(10)
        a = slice(None)
        np.testing.assert_array_equal(njit(test_impl2, parallel=True)(A, a), test_impl2(A, a))

    @skip_unsupported
    def test_slice_dtype_issue_5056(self):

        @njit(parallel=True)
        def test_impl(data):
            N = data.shape[0]
            sums = np.zeros(N)
            for i in prange(N):
                sums[i] = np.sum(data[np.int32(0):np.int32(1)])
            return sums
        data = np.arange(10.0)
        np.testing.assert_array_equal(test_impl(data), test_impl.py_func(data))

    @skip_unsupported
    def test_global_tuple(self):
        """make sure a global tuple with non-integer values does not cause errors
        (test for #6726).
        """

        def test_impl():
            d = GVAL[0]
            return d
        self.assertEqual(njit(test_impl, parallel=True)(), test_impl())