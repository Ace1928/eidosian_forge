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
class TestParforBasic(TestParforsBase):
    """Smoke tests for the parfors transforms. These tests check the most basic
    functionality"""

    def __init__(self, *args):
        TestParforsBase.__init__(self, *args)
        m = np.reshape(np.arange(12.0), (3, 4))
        self.simple_args = [np.arange(3.0), np.arange(4.0), m, m.T]

    def test_simple01(self):

        def test_impl():
            return np.ones(())
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_simple02(self):

        def test_impl():
            return np.ones((1,))
        self.check(test_impl)

    def test_simple03(self):

        def test_impl():
            return np.ones((1, 2))
        self.check(test_impl)

    def test_simple04(self):

        def test_impl():
            return np.ones(1)
        self.check(test_impl)

    def test_simple07(self):

        def test_impl():
            return np.ones((1, 2), dtype=np.complex128)
        self.check(test_impl)

    def test_simple08(self):

        def test_impl():
            return np.ones((1, 2)) + np.ones((1, 2))
        self.check(test_impl)

    def test_simple09(self):

        def test_impl():
            return np.ones((1, 1))
        self.check(test_impl)

    def test_simple10(self):

        def test_impl():
            return np.ones((0, 0))
        self.check(test_impl)

    def test_simple11(self):

        def test_impl():
            return np.ones((10, 10)) + 1.0
        self.check(test_impl)

    def test_simple12(self):

        def test_impl():
            return np.ones((10, 10)) + np.complex128(1.0)
        self.check(test_impl)

    def test_simple13(self):

        def test_impl():
            return np.complex128(1.0)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_simple14(self):

        def test_impl():
            return np.ones((10, 10))[0::20]
        self.check(test_impl)

    def test_simple15(self):

        def test_impl(v1, v2, m1, m2):
            return v1 + v1
        self.check(test_impl, *self.simple_args)

    def test_simple16(self):

        def test_impl(v1, v2, m1, m2):
            return m1 + m1
        self.check(test_impl, *self.simple_args)

    def test_simple17(self):

        def test_impl(v1, v2, m1, m2):
            return m2 + v1
        self.check(test_impl, *self.simple_args)

    @needs_lapack
    def test_simple18(self):

        def test_impl(v1, v2, m1, m2):
            return m1.T + np.linalg.svd(m2)[1]
        self.check(test_impl, *self.simple_args)

    @needs_blas
    def test_simple19(self):

        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, v2)
        self.check(test_impl, *self.simple_args)

    @needs_blas
    def test_simple20(self):

        def test_impl(v1, v2, m1, m2):
            return np.dot(m1, m2)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, *self.simple_args)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    @needs_blas
    def test_simple21(self):

        def test_impl(v1, v2, m1, m2):
            return np.dot(v1, v1)
        self.check(test_impl, *self.simple_args)

    def test_simple22(self):

        def test_impl(v1, v2, m1, m2):
            return np.sum(v1 + v1)
        self.check(test_impl, *self.simple_args)

    def test_simple23(self):

        def test_impl(v1, v2, m1, m2):
            x = 2 * v1
            y = 2 * v1
            return 4 * np.sum(x ** 2 + y ** 2 < 1) / 10
        self.check(test_impl, *self.simple_args)

    def test_simple24(self):

        def test_impl():
            n = 20
            A = np.ones((n, n))
            b = np.arange(n)
            return np.sum(A[:, b])
        self.check(test_impl)

    @disabled_test
    def test_simple_operator_15(self):
        """same as corresponding test_simple_<n> case but using operator.add"""

        def test_impl(v1, v2, m1, m2):
            return operator.add(v1, v1)
        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_16(self):

        def test_impl(v1, v2, m1, m2):
            return operator.add(m1, m1)
        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_17(self):

        def test_impl(v1, v2, m1, m2):
            return operator.add(m2, v1)
        self.check(test_impl, *self.simple_args)

    def test_inplace_alias(self):

        def test_impl(a):
            a += 1
            a[:] = 3

        def comparer(a, b):
            np.testing.assert_equal(a, b)
        x = np.ones(1)
        self.check(test_impl, x, check_arg_equality=[comparer])