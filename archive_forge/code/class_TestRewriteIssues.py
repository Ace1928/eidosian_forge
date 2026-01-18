import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
class TestRewriteIssues(MemoryLeakMixin, TestCase):

    def test_issue_1184(self):
        from numba import jit
        import numpy as np

        @jit(nopython=True)
        def foo(arr):
            return arr

        @jit(nopython=True)
        def bar(arr):
            c = foo(arr)
            d = foo(arr)
            return (c, d)
        arr = np.arange(10)
        out_c, out_d = bar(arr)
        self.assertIs(out_c, out_d)
        self.assertIs(out_c, arr)

    def test_issue_1264(self):
        n = 100
        x = np.random.uniform(size=n * 3).reshape((n, 3))
        expected = distance_matrix(x)
        actual = njit(distance_matrix)(x)
        np.testing.assert_array_almost_equal(expected, actual)
        gc.collect()

    def test_issue_1372(self):
        """Test array expression with duplicated term"""
        from numba import njit

        @njit
        def foo(a, b):
            b = np.sin(b)
            return b + b + a
        a = np.random.uniform(10)
        b = np.random.uniform(10)
        expect = foo.py_func(a, b)
        got = foo(a, b)
        np.testing.assert_allclose(got, expect)

    def test_unary_arrayexpr(self):
        """
        Typing of unary array expression (np.negate) can be incorrect.
        """

        @njit
        def foo(a, b):
            return b - a + -a
        b = 1.5
        a = np.arange(10, dtype=np.int32)
        expect = foo.py_func(a, b)
        got = foo(a, b)
        self.assertPreciseEqual(got, expect)

    def test_bitwise_arrayexpr(self):
        """
        Typing of bitwise boolean array expression can be incorrect
        (issue #1813).
        """

        @njit
        def foo(a, b):
            return ~(a & ~b)
        a = np.array([True, True, False, False])
        b = np.array([False, True, False, True])
        expect = foo.py_func(a, b)
        got = foo(a, b)
        self.assertPreciseEqual(got, expect)

    def test_annotations(self):
        """
        Type annotation of array expressions with disambiguated
        variable names (issue #1466).
        """
        cfunc = njit(variable_name_reuse)
        a = np.linspace(0, 1, 10)
        cfunc(a, a, a, a)
        buf = StringIO()
        cfunc.inspect_types(buf)
        res = buf.getvalue()
        self.assertIn('#   u.1 = ', res)
        self.assertIn('#   u.2 = ', res)

    def test_issue_5599_name_collision(self):

        @njit
        def f(x):
            arr = np.ones(x)
            for _ in range(2):
                val = arr * arr
                arr = arr.copy()
            return arr
        got = f(5)
        expect = f.py_func(5)
        np.testing.assert_array_equal(got, expect)