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
class TestListComprehension(TestCase):

    def test_comp_list(self):
        pyfunc = comp_list
        cfunc = njit((types.intp,))(pyfunc)
        self.assertEqual(cfunc(5), pyfunc(5))
        self.assertEqual(cfunc(0), pyfunc(0))
        self.assertEqual(cfunc(-1), pyfunc(-1))

    def test_bulk_use_cases(self):
        """ Tests the large number of use cases defined below """

        @jit(nopython=True)
        def fib3(n):
            if n < 2:
                return n
            return fib3(n - 1) + fib3(n - 2)

        def list1(x):
            """ Test basic list comprehension """
            return [i for i in range(1, len(x) - 1)]

        def list2(x):
            """ Test conditional list comprehension """
            return [y for y in x if y < 2]

        def list3(x):
            """ Test ternary list comprehension """
            return [y if y < 2 else -1 for y in x]

        def list4(x):
            """ Test list comprehension to np.array ctor """
            return np.array([1, 2, 3])

        def list5(x):
            """ Test nested list comprehension to np.array ctor """
            return np.array([np.array([z for z in x]) for y in x])

        def list6(x):
            """ Test use of inner function in list comprehension """

            def inner(x):
                return x + 1
            return [inner(z) for z in x]

        def list7(x):
            """ Test use of closure in list comprehension """
            y = 3

            def inner(x):
                return x + y
            return [inner(z) for z in x]

        def list8(x):
            """ Test use of list comprehension as arg to inner function """
            l = [z + 1 for z in x]

            def inner(x):
                return x[0] + 1
            q = inner(l)
            return q

        def list9(x):
            """ Test use of list comprehension access in closure """
            l = [z + 1 for z in x]

            def inner(x):
                return x[0] + l[1]
            return inner(x)

        def list10(x):
            """ Test use of list comprehension access in closure and as arg """
            l = [z + 1 for z in x]

            def inner(x):
                return [y + l[0] for y in x]
            return inner(l)

        def list11(x):
            """ Test scalar array construction in list comprehension """
            l = [np.array(z) for z in x]
            return l

        def list12(x):
            """ Test scalar type conversion construction in list comprehension """
            l = [np.float64(z) for z in x]
            return l

        def list13(x):
            """ Test use of explicit numpy scalar ctor reference in list comprehension """
            l = [numpy.float64(z) for z in x]
            return l

        def list14(x):
            """ Test use of python scalar ctor reference in list comprehension """
            l = [float(z) for z in x]
            return l

        def list15(x):
            """ Test use of python scalar ctor reference in list comprehension followed by np array construction from the list"""
            l = [float(z) for z in x]
            return np.array(l)

        def list16(x):
            """ Test type unification from np array ctors consuming list comprehension """
            l1 = [float(z) for z in x]
            l2 = [z for z in x]
            ze = np.array(l1)
            oe = np.array(l2)
            return ze + oe

        def list17(x):
            """ Test complex list comprehension including math calls """
            return [(a, b, c) for a in x for b in x for c in x if np.sqrt(a ** 2 + b ** 2) == c]
        _OUTER_SCOPE_VAR = 9

        def list18(x):
            """ Test loop list with outer scope var as conditional"""
            z = []
            for i in x:
                if i < _OUTER_SCOPE_VAR:
                    z.append(i)
            return z
        _OUTER_SCOPE_VAR = 9

        def list19(x):
            """ Test list comprehension with outer scope as conditional"""
            return [i for i in x if i < _OUTER_SCOPE_VAR]

        def list20(x):
            """ Test return empty list """
            return [i for i in x if i == -1000]

        def list21(x):
            """ Test call a jitted function in a list comprehension """
            return [fib3(i) for i in x]

        def list22(x):
            """ Test create two lists comprehensions and a third walking the first two """
            a = [y - 1 for y in x]
            b = [y + 1 for y in x]
            return [x for x in a for y in b if x == y]

        def list23(x):
            """ Test operation on comprehension generated list """
            z = [y for y in x]
            z.append(1)
            return z

        def list24(x):
            """ Test type promotion """
            z = [float(y) if y > 3 else y for y in x]
            return z

        def list25(x):
            included = np.array([1, 2, 6, 8])
            not_included = [i for i in range(10) if i not in list(included)]
            return not_included
        f = [list1, list2, list3, list4, list6, list7, list8, list9, list10, list11, list12, list13, list14, list15, list16, list17, list18, list19, list20, list21, list22, list23, list24, list25]
        var = [1, 2, 3, 4, 5]
        for ref in f:
            try:
                cfunc = jit(nopython=True)(ref)
                self.assertEqual(cfunc(var), ref(var))
            except ValueError:
                try:
                    np.testing.assert_allclose(cfunc(var), ref(var))
                except Exception:
                    raise
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(list5)
            cfunc(var)
        if sys.maxsize > 2 ** 32:
            bits = 64
        else:
            bits = 32

    def test_objmode_inlining(self):

        def objmode_func(y):
            z = object()
            inlined = [x for x in y]
            return inlined
        cfunc = jit(forceobj=True)(objmode_func)
        t = [1, 2, 3]
        expected = objmode_func(t)
        got = cfunc(t)
        self.assertPreciseEqual(expected, got)