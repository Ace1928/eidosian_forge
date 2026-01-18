import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
@skip_parfors_unsupported
class TestParforNumPy(TestParforsBase):
    """Tests NumPy functionality under parfors"""

    @needs_blas
    def test_mvdot(self):

        def test_impl(a, v):
            return np.dot(a, v)
        A = np.linspace(0, 1, 20).reshape(2, 10)
        v = np.linspace(2, 1, 10)
        self.check(test_impl, A, v)

    def test_fuse_argmin_argmax_max_min(self):
        for op in [np.argmin, np.argmax, np.min, np.max]:

            def test_impl(n):
                A = np.ones(n)
                C = op(A)
                B = A.sum()
                return B + C
            self.check(test_impl, 256)
            self.assertEqual(countParfors(test_impl, (types.int64,)), 1)
            self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    def test_np_random_func_direct_import(self):

        def test_impl(n):
            A = randn(n)
            return A[0]
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)

    def test_arange(self):

        def test_impl1(n):
            return np.arange(n)

        def test_impl2(s, n):
            return np.arange(s, n)

        def test_impl3(s, n, t):
            return np.arange(s, n, t)
        for arg in [11, 128, 30.0, complex(4, 5), complex(5, 4)]:
            self.check(test_impl1, arg)
            self.check(test_impl2, 2, arg)
            self.check(test_impl3, 2, arg, 2)

    def test_arange_dtype(self):

        def test_impl1(n):
            return np.arange(n, dtype=np.float32)

        def test_impl2(s, n):
            return np.arange(s, n, dtype=np.float32)

        def test_impl3(s, n, t):
            return np.arange(s, n, t, dtype=np.float32)
        for arg in [11, 128, 30.0]:
            self.check(test_impl1, arg)
            self.check(test_impl2, 2, arg)
            self.check(test_impl3, 2, arg, 2)

    def test_linspace(self):

        def test_impl1(start, stop):
            return np.linspace(start, stop)

        def test_impl2(start, stop, num):
            return np.linspace(start, stop, num)
        for arg in [11, 128, 30.0, complex(4, 5), complex(5, 4)]:
            self.check(test_impl1, 2, arg)
            self.check(test_impl2, 2, arg, 30)

    def test_mean(self):

        def test_impl(A):
            return A.mean()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'),)), 1)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 2, 'C'),)), 1)
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_var(self):

        def test_impl(A):
            return A.var()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        C = A + 1j * A
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.check(test_impl, C)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'),)), 2)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 2, 'C'),)), 2)
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_std(self):

        def test_impl(A):
            return A.std()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        C = A + 1j * A
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.check(test_impl, C)
        argty = (types.Array(types.float64, 1, 'C'),)
        self.assertEqual(countParfors(test_impl, argty), 2)
        self.assertEqual(countParfors(test_impl, argty), 2)
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_random_parfor(self):
        """
        Test function with only a random call to make sure a random function
        like ranf is actually translated to a parfor.
        """

        def test_impl(n):
            A = np.random.ranf((n, n))
            return A
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)

    def test_randoms(self):

        def test_impl(n):
            A = np.random.standard_normal(size=(n, n))
            B = np.random.randn(n, n)
            C = np.random.normal(0.0, 1.0, (n, n))
            D = np.random.chisquare(1.0, (n, n))
            E = np.random.randint(1, high=3, size=(n, n))
            F = np.random.triangular(1, 2, 3, (n, n))
            return np.sum(A + B + C + D + E + F)
        n = 128
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
        parfor_output = cpfunc.entry_point(n)
        py_output = test_impl(n)
        np.testing.assert_allclose(parfor_output, py_output, rtol=0.05)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)

    def test_dead_randoms(self):

        def test_impl(n):
            A = np.random.standard_normal(size=(n, n))
            B = np.random.randn(n, n)
            C = np.random.normal(0.0, 1.0, (n, n))
            D = np.random.chisquare(1.0, (n, n))
            E = np.random.randint(1, high=3, size=(n, n))
            F = np.random.triangular(1, 2, 3, (n, n))
            return 3
        n = 128
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
        parfor_output = cpfunc.entry_point(n)
        py_output = test_impl(n)
        self.assertEqual(parfor_output, py_output)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 0)

    def test_min(self):

        def test_impl1(A):
            return A.min()

        def test_impl2(A):
            return np.min(A)
        n = 211
        A = np.random.ranf(n)
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([np.inf, np.inf])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'zero-size array to reduction operation minimum which has no identity'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_max(self):

        def test_impl1(A):
            return A.max()

        def test_impl2(A):
            return np.max(A)
        n = 211
        A = np.random.ranf(n)
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([-np.inf, -np.inf])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'zero-size array to reduction operation maximum which has no identity'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_argmax(self):

        def test_impl1(A):
            return A.argmax()

        def test_impl2(A):
            return np.argmax(A)
        n = 211
        A = np.array([1.0, 0.0, 3.0, 2.0, 3.0])
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([1.0, 0.0, np.nan, 2.0, 3.0])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'attempt to get argmax of an empty sequence'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_argmin(self):

        def test_impl1(A):
            return A.argmin()

        def test_impl2(A):
            return np.argmin(A)
        n = 211
        A = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([1.0, 0.0, np.nan, 0.0, 3.0])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'attempt to get argmin of an empty sequence'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda: self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_ndarray_fill(self):

        def test_impl(x):
            x.fill(7.0)
            return x
        x = np.zeros(10)
        self.check(test_impl, x)
        argty = (types.Array(types.float64, 1, 'C'),)
        self.assertEqual(countParfors(test_impl, argty), 1)

    def test_ndarray_fill2d(self):

        def test_impl(x):
            x.fill(7.0)
            return x
        x = np.zeros((2, 2))
        self.check(test_impl, x)
        argty = (types.Array(types.float64, 2, 'C'),)
        self.assertEqual(countParfors(test_impl, argty), 1)

    def test_reshape_with_neg_one(self):

        def test_impl(a, b):
            result_matrix = np.zeros((b, b, 1), dtype=np.float64)
            sub_a = a[0:b]
            a = sub_a.size
            b = a / 1
            z = sub_a.reshape(-1, 1)
            result_data = sub_a / z
            result_matrix[:, :, 0] = result_data
            return result_matrix
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        b = 3
        self.check(test_impl, a, b)

    def test_reshape_with_large_neg(self):

        def test_impl(a, b):
            result_matrix = np.zeros((b, b, 1), dtype=np.float64)
            sub_a = a[0:b]
            a = sub_a.size
            b = a / 1
            z = sub_a.reshape(-1307, 1)
            result_data = sub_a / z
            result_matrix[:, :, 0] = result_data
            return result_matrix
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        b = 3
        self.check(test_impl, a, b)

    def test_reshape_with_too_many_neg_one(self):
        with self.assertRaises(errors.UnsupportedRewriteError) as raised:

            @njit(parallel=True)
            def test_impl(a, b):
                rm = np.zeros((b, b, 1), dtype=np.float64)
                sub_a = a[0:b]
                a = sub_a.size
                b = a / 1
                z = sub_a.reshape(-1, -1)
                result_data = sub_a / z
                rm[:, :, 0] = result_data
                return rm
            a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            b = 3
            test_impl(a, b)
        msg = 'The reshape API may only include one negative argument.'
        self.assertIn(msg, str(raised.exception))

    def test_0d_array(self):

        def test_impl(n):
            return np.sum(n) + np.prod(n) + np.min(n) + np.max(n) + np.var(n)
        self.check(test_impl, np.array(7), check_scheduling=False)

    def test_real_imag_attr(self):

        def test_impl(z):
            return np.sum(z.real ** 2 + z.imag ** 2)
        z = np.arange(5) * (1 + 1j)
        self.check(test_impl, z)
        self.assertEqual(countParfors(test_impl, (types.complex128[::1],)), 1)