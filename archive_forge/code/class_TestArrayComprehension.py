import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
class TestArrayComprehension(unittest.TestCase):
    _numba_parallel_test_ = False

    def check(self, pyfunc, *args, **kwargs):
        """A generic check function that run both pyfunc, and jitted pyfunc,
        and compare results."""
        run_parallel = kwargs.get('run_parallel', False)
        assert_allocate_list = kwargs.get('assert_allocate_list', False)
        assert_dtype = kwargs.get('assert_dtype', False)
        cfunc = jit(nopython=True, parallel=run_parallel)(pyfunc)
        pyres = pyfunc(*args)
        cres = cfunc(*args)
        np.testing.assert_array_equal(pyres, cres)
        if assert_dtype:
            self.assertEqual(cres[1].dtype, assert_dtype)
        if assert_allocate_list:
            self.assertIn('allocate list', cfunc.inspect_llvm(cfunc.signatures[0]))
        else:
            self.assertNotIn('allocate list', cfunc.inspect_llvm(cfunc.signatures[0]))
        if run_parallel:
            self.assertIn('@do_scheduling', cfunc.inspect_llvm(cfunc.signatures[0]))

    def test_comp_with_array_1(self):

        def comp_with_array_1(n):
            m = n * 2
            l = np.array([i + m for i in range(n)])
            return l
        self.check(comp_with_array_1, 5)
        if PARALLEL_SUPPORTED:
            self.check(comp_with_array_1, 5, run_parallel=True)

    def test_comp_with_array_2(self):

        def comp_with_array_2(n, threshold):
            A = np.arange(-n, n)
            return np.array([x * x if x < threshold else x * 2 for x in A])
        self.check(comp_with_array_2, 5, 0)

    def test_comp_with_array_noinline(self):

        def comp_with_array_noinline(n):
            m = n * 2
            l = np.array([i + m for i in range(n)])
            return l
        import numba.core.inline_closurecall as ic
        try:
            ic.enable_inline_arraycall = False
            self.check(comp_with_array_noinline, 5, assert_allocate_list=True)
        finally:
            ic.enable_inline_arraycall = True

    def test_comp_with_array_noinline_issue_6053(self):

        def comp_with_array_noinline(n):
            lst = [0]
            for i in range(n):
                lst.append(i)
            l = np.array(lst)
            return l
        self.check(comp_with_array_noinline, 5, assert_allocate_list=True)

    def test_comp_nest_with_array(self):

        def comp_nest_with_array(n):
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        self.check(comp_nest_with_array, 5)
        if PARALLEL_SUPPORTED:
            self.check(comp_nest_with_array, 5, run_parallel=True)

    def test_comp_nest_with_array_3(self):

        def comp_nest_with_array_3(n):
            l = np.array([[[i * j * k for k in range(n)] for j in range(n)] for i in range(n)])
            return l
        self.check(comp_nest_with_array_3, 5)
        if PARALLEL_SUPPORTED:
            self.check(comp_nest_with_array_3, 5, run_parallel=True)

    def test_comp_nest_with_array_noinline(self):

        def comp_nest_with_array_noinline(n):
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        import numba.core.inline_closurecall as ic
        try:
            ic.enable_inline_arraycall = False
            self.check(comp_nest_with_array_noinline, 5, assert_allocate_list=True)
        finally:
            ic.enable_inline_arraycall = True

    def test_comp_with_array_range(self):

        def comp_with_array_range(m, n):
            l = np.array([i for i in range(m, n)])
            return l
        self.check(comp_with_array_range, 5, 10)

    def test_comp_with_array_range_and_step(self):

        def comp_with_array_range_and_step(m, n):
            l = np.array([i for i in range(m, n, 2)])
            return l
        self.check(comp_with_array_range_and_step, 5, 10)

    def test_comp_with_array_conditional(self):

        def comp_with_array_conditional(n):
            l = np.array([i for i in range(n) if i % 2 == 1])
            return l
        self.check(comp_with_array_conditional, 10, assert_allocate_list=True)

    def test_comp_nest_with_array_conditional(self):

        def comp_nest_with_array_conditional(n):
            l = np.array([[i * j for j in range(n)] for i in range(n) if i % 2 == 1])
            return l
        self.check(comp_nest_with_array_conditional, 5, assert_allocate_list=True)

    @unittest.skipUnless(numpy_version < (1, 24), 'Setting an array element with a sequence is removed in NumPy 1.24')
    def test_comp_nest_with_dependency(self):

        def comp_nest_with_dependency(n):
            l = np.array([[i * j for j in range(i + 1)] for i in range(n)])
            return l
        with self.assertRaises(TypingError) as raises:
            self.check(comp_nest_with_dependency, 5)
        self.assertIn(_header_lead, str(raises.exception))
        self.assertIn('array(undefined,', str(raises.exception))

    def test_comp_unsupported_iter(self):

        def comp_unsupported_iter():
            val = zip([1, 2, 3], [4, 5, 6])
            return np.array([a for a, b in val])
        with self.assertRaises(TypingError) as raises:
            self.check(comp_unsupported_iter)
        self.assertIn(_header_lead, str(raises.exception))
        self.assertIn('Unsupported iterator found in array comprehension', str(raises.exception))

    def test_no_array_comp(self):

        def no_array_comp1(n):
            l = [1, 2, 3, 4]
            a = np.array(l)
            return a
        self.check(no_array_comp1, 10, assert_allocate_list=False)

        def no_array_comp2(n):
            l = [1, 2, 3, 4]
            a = np.array(l)
            l.append(5)
            return a
        self.check(no_array_comp2, 10, assert_allocate_list=True)

    def test_nested_array(self):

        def nested_array(n):
            l = np.array([np.array([x for x in range(n)]) for y in range(n)])
            return l
        self.check(nested_array, 10)

    def test_nested_array_with_const(self):

        def nested_array(n):
            l = np.array([np.array([x for x in range(3)]) for y in range(4)])
            return l
        self.check(nested_array, 0)

    def test_array_comp_with_iter(self):

        def array_comp(a):
            l = np.array([x * x for x in a])
            return l
        l = [1, 2, 3, 4, 5]
        self.check(array_comp, l)
        self.check(array_comp, np.array(l))
        self.check(array_comp, tuple(l))
        self.check(array_comp, typed.List(l))

    def test_array_comp_with_dtype(self):

        def array_comp(n):
            l = np.array([i for i in range(n)], dtype=np.complex64)
            return l
        self.check(array_comp, 10, assert_dtype=np.complex64)

    def test_array_comp_inferred_dtype(self):

        def array_comp(n):
            l = np.array([i * 1j for i in range(n)])
            return l
        self.check(array_comp, 10)

    def test_array_comp_inferred_dtype_nested(self):

        def array_comp(n):
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        self.check(array_comp, 10)

    def test_array_comp_inferred_dtype_nested_sum(self):

        def array_comp(n):
            l = np.array([[i * j for j in range(n)] for i in range(n)])
            return l
        self.check(array_comp, 10)

    def test_array_comp_inferred_dtype_outside_setitem(self):

        def array_comp(n, v):
            arr = np.array([i for i in range(n)])
            arr[0] = v
            return arr
        v = 1.2
        self.check(array_comp, 10, v, assert_dtype=np.intp)
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(array_comp)
            cfunc(10, 2.3j)
        self.assertIn(_header_lead + ' Function({})'.format(operator.setitem), str(raises.exception))
        self.assertIn('(array({}, 1d, C), Literal[int](0), complex128)'.format(types.intp), str(raises.exception))

    def test_array_comp_shuffle_sideeffect(self):
        nelem = 100

        @jit(nopython=True)
        def foo():
            numbers = np.array([i for i in range(nelem)])
            np.random.shuffle(numbers)
            print(numbers)
        with captured_stdout() as gotbuf:
            foo()
        got = gotbuf.getvalue().strip()
        with captured_stdout() as expectbuf:
            print(np.array([i for i in range(nelem)]))
        expect = expectbuf.getvalue().strip()
        self.assertNotEqual(got, expect)
        self.assertRegex(got, '\\[(\\s*\\d+)+\\]')

    def test_empty_list_not_removed(self):

        def f(x):
            t = []
            myList = np.array([1])
            a = np.random.choice(myList, 1)
            t.append(x + a)
            return a
        self.check(f, 5, assert_allocate_list=True)

    def test_reuse_of_array_var(self):
        """ Test issue 3742 """

        def foo(n):
            [i for i in range(1)]
            z = np.empty(n)
            for i in range(n):
                z = np.zeros(n)
                z[i] = i
            return z
        self.check(foo, 10, assert_allocate_list=True)