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
class TestParforsSlice(TestParforsBase):

    def test_parfor_slice1(self):

        def test_impl(a):
            n, = a.shape
            b = a[0:n - 2] + a[1:n - 1]
            return b
        self.check(test_impl, np.ones(10))

    def test_parfor_slice2(self):

        def test_impl(a, m):
            n, = a.shape
            b = a[0:n - 2] + a[1:m]
            return b
        self.check(test_impl, np.ones(10), 9)
        with self.assertRaises(AssertionError) as raises:
            njit(parallel=True)(test_impl)(np.ones(10), 10)
        self.assertIn('do not match', str(raises.exception))

    def test_parfor_slice3(self):

        def test_impl(a):
            m, n = a.shape
            b = a[0:m - 1, 0:n - 1] + a[1:m, 1:n]
            return b
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice4(self):

        def test_impl(a):
            m, n = a.shape
            b = a[:, 0:n - 1] + a[:, 1:n]
            return b
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice5(self):

        def test_impl(a):
            m, n = a.shape
            b = a[0:m - 1, :] + a[1:m, :]
            return b
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice6(self):

        def test_impl(a):
            b = a.transpose()
            c = a[1, :] + b[:, 1]
            return c
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice7(self):

        def test_impl(a):
            b = a.transpose()
            c = a[1, :] + b[1, :]
            return c
        self.check(test_impl, np.ones((3, 3)))
        with self.assertRaises(AssertionError) as raises:
            njit(parallel=True)(test_impl)(np.ones((3, 4)))
        self.assertIn('do not match', str(raises.exception))

    @disabled_test
    def test_parfor_slice8(self):

        def test_impl(a):
            m, n = a.shape
            b = a.transpose()
            b[1:m, 1:n] = a[1:m, 1:n]
            return b
        self.check(test_impl, np.arange(9).reshape((3, 3)))

    @disabled_test
    def test_parfor_slice9(self):

        def test_impl(a):
            m, n = a.shape
            b = a.transpose()
            b[1:n, 1:m] = a[:, 1:m]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    @disabled_test
    def test_parfor_slice10(self):

        def test_impl(a):
            m, n = a.shape
            b = a.transpose()
            b[2, 1:m] = a[2, 1:m]
            return b
        self.check(test_impl, np.arange(9).reshape((3, 3)))

    def test_parfor_slice11(self):

        def test_impl(a):
            m, n, l = a.shape
            b = a.copy()
            b[:, 1, 1:l] = a[:, 2, 1:l]
            return b
        self.check(test_impl, np.arange(27).reshape((3, 3, 3)))

    def test_parfor_slice12(self):

        def test_impl(a):
            m, n = a.shape
            b = a.copy()
            b[1, 1:-1] = a[0, :-2]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    def test_parfor_slice13(self):

        def test_impl(a):
            m, n = a.shape
            b = a.copy()
            c = -1
            b[1, 1:c] = a[0, -n:c - 1]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    def test_parfor_slice14(self):

        def test_impl(a):
            m, n = a.shape
            b = a.copy()
            b[1, :-1] = a[0, -3:4]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    def test_parfor_slice15(self):

        def test_impl(a):
            m, n = a.shape
            b = a.copy()
            b[1, -(n - 1):] = a[0, -3:4]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    @disabled_test
    def test_parfor_slice16(self):
        """ This test is disabled because if n is larger than the array size
            then n and n-1 will both be the end of the array and thus the
            slices will in fact be of different sizes and unable to fuse.
        """

        def test_impl(a, b, n):
            assert a.shape == b.shape
            a[1:n] = 10
            b[0:n - 1] = 10
            return a * b
        self.check(test_impl, np.ones(10), np.zeros(10), 8)
        args = (numba.float64[:], numba.float64[:], numba.int64)
        self.assertEqual(countParfors(test_impl, args), 2)

    def test_parfor_slice17(self):

        def test_impl(m, A):
            B = np.zeros(m)
            n = len(A)
            B[-n:] = A
            return B
        self.check(test_impl, 10, np.ones(10))

    def test_parfor_slice18(self):

        def test_impl():
            a = np.zeros(10)
            a[1:8] = np.arange(0, 7)
            y = a[3]
            return y
        self.check(test_impl)

    def test_parfor_slice19(self):

        def test_impl(X):
            X[:0] += 1
            return X
        self.check(test_impl, np.ones(10))

    def test_parfor_slice20(self):

        def test_impl():
            a = np.ones(10)
            c = a[1:]
            s = len(c)
            return s
        self.check(test_impl, check_scheduling=False)

    def test_parfor_slice21(self):

        def test_impl(x1, x2):
            x1 = x1.reshape(x1.size, 1)
            x2 = x2.reshape(x2.size, 1)
            return x1 >= x2[:-1, :]
        x1 = np.random.rand(5)
        x2 = np.random.rand(6)
        self.check(test_impl, x1, x2)

    def test_parfor_slice22(self):

        def test_impl(x1, x2):
            b = np.zeros((10,))
            for i in prange(1):
                b += x1[:, x2]
            return b
        x1 = np.zeros((10, 7))
        x2 = np.array(4)
        self.check(test_impl, x1, x2)

    def test_parfor_slice23(self):

        def test_impl(x):
            x[:0] = 2
            return x
        self.check(test_impl, np.ones(10))

    def test_parfor_slice24(self):

        def test_impl(m, A, n):
            B = np.zeros(m)
            C = B[n:]
            C = A[:len(C)]
            return B
        for i in range(-15, 15):
            self.check(test_impl, 10, np.ones(10), i)

    def test_parfor_slice25(self):

        def test_impl(m, A, n):
            B = np.zeros(m)
            C = B[:n]
            C = A[:len(C)]
            return B
        for i in range(-15, 15):
            self.check(test_impl, 10, np.ones(10), i)

    def test_parfor_slice26(self):

        def test_impl(a):
            n, = a.shape
            b = a.copy()
            b[-(n - 1):] = a[-3:4]
            return b
        self.check(test_impl, np.arange(4))

    def test_parfor_slice27(self):

        def test_impl(a):
            n_valid_vals = 0
            for i in prange(a.shape[0]):
                if a[i] != 0:
                    n_valid_vals += 1
                if n_valid_vals:
                    unused = a[:n_valid_vals]
            return 0
        self.check(test_impl, np.arange(3))

    def test_parfor_array_access_lower_slice(self):
        for ts in [slice(1, 3, None), slice(2, None, None), slice(None, 2, -1), slice(None, None, None), slice(None, None, -2)]:

            def test_impl(n):
                X = np.arange(n * 4).reshape((n, 4))
                y = 0
                for i in numba.prange(n):
                    y += X[i, ts].sum()
                return y
            n = 10
            self.check(test_impl, n)
            X = np.arange(n * 4).reshape((n, 4))

            def test_impl(X):
                y = 0
                for i in numba.prange(X.shape[0]):
                    y += X[i, ts].sum()
                return y
            self.check(test_impl, X)