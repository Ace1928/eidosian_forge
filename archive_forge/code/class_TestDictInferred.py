import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestDictInferred(TestCase):

    def test_simple_literal(self):

        @njit
        def foo():
            d = Dict()
            d[123] = 321
            return d
        k, v = (123, 321)
        d = foo()
        self.assertEqual(dict(d), {k: v})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_simple_args(self):

        @njit
        def foo(k, v):
            d = Dict()
            d[k] = v
            return d
        k, v = (123, 321)
        d = foo(k, v)
        self.assertEqual(dict(d), {k: v})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_simple_upcast(self):

        @njit
        def foo(k, v, w):
            d = Dict()
            d[k] = v
            d[k] = w
            return d
        k, v, w = (123, 32.1, 321)
        d = foo(k, v, w)
        self.assertEqual(dict(d), {k: w})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_conflicting_value_type(self):

        @njit
        def foo(k, v, w):
            d = Dict()
            d[k] = v
            d[k] = w
            return d
        k, v, w = (123, 321, 32.1)
        with self.assertRaises(TypingError) as raises:
            foo(k, v, w)
        self.assertIn('cannot safely cast float64 to {}'.format(typeof(v)), str(raises.exception))

    def test_conflicting_key_type(self):

        @njit
        def foo(k, h, v):
            d = Dict()
            d[k] = v
            d[h] = v
            return d
        k, h, v = (123, 123.1, 321)
        with self.assertRaises(TypingError) as raises:
            foo(k, h, v)
        self.assertIn('cannot safely cast float64 to {}'.format(typeof(v)), str(raises.exception))

    def test_conflict_key_type_non_number(self):

        @njit
        def foo(k1, v1, k2):
            d = Dict()
            d[k1] = v1
            return (d, d[k2])
        k1 = (np.int8(1), np.int8(2))
        k2 = (np.int32(1), np.int32(2))
        v1 = np.intp(123)
        with warnings.catch_warnings(record=True) as w:
            d, dk2 = foo(k1, v1, k2)
        self.assertEqual(len(w), 1)
        msg = 'unsafe cast from UniTuple(int32 x 2) to UniTuple(int8 x 2)'
        self.assertIn(msg, str(w[0]))
        keys = list(d.keys())
        self.assertEqual(keys[0], (1, 2))
        self.assertEqual(dk2, d[np.int32(1), np.int32(2)])

    def test_ifelse_filled_both_branches(self):

        @njit
        def foo(k, v):
            d = Dict()
            if k:
                d[k] = v
            else:
                d[57005] = v + 1
            return d
        k, v = (123, 321)
        d = foo(k, v)
        self.assertEqual(dict(d), {k: v})
        k, v = (0, 0)
        d = foo(k, v)
        self.assertEqual(dict(d), {57005: v + 1})

    def test_ifelse_empty_one_branch(self):

        @njit
        def foo(k, v):
            d = Dict()
            if k:
                d[k] = v
            return d
        k, v = (123, 321)
        d = foo(k, v)
        self.assertEqual(dict(d), {k: v})
        k, v = (0, 0)
        d = foo(k, v)
        self.assertEqual(dict(d), {})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_loop(self):

        @njit
        def foo(ks, vs):
            d = Dict()
            for k, v in zip(ks, vs):
                d[k] = v
            return d
        vs = list(range(4))
        ks = list(map(lambda x: x + 100, vs))
        d = foo(ks, vs)
        self.assertEqual(dict(d), dict(zip(ks, vs)))

    def test_unused(self):

        @njit
        def foo():
            d = Dict()
            return d
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('imprecise type', str(raises.exception))

    def test_define_after_use(self):

        @njit
        def foo(define):
            d = Dict()
            ct = len(d)
            for k, v in d.items():
                ct += v
            if define:
                d[1] = 2
            return (ct, d, len(d))
        ct, d, n = foo(True)
        self.assertEqual(ct, 0)
        self.assertEqual(n, 1)
        self.assertEqual(dict(d), {1: 2})
        ct, d, n = foo(False)
        self.assertEqual(ct, 0)
        self.assertEqual(dict(d), {})
        self.assertEqual(n, 0)

    def test_dict_of_dict(self):

        @njit
        def foo(k1, k2, v):
            d = Dict()
            z1 = Dict()
            z1[k1 + 1] = v + k1
            z2 = Dict()
            z2[k2 + 2] = v + k2
            d[k1] = z1
            d[k2] = z2
            return d
        k1, k2, v = (100, 200, 321)
        d = foo(k1, k2, v)
        self.assertEqual(dict(d), {k1: {k1 + 1: k1 + v}, k2: {k2 + 2: k2 + v}})

    def test_comprehension_basic(self):

        @njit
        def foo():
            return {i: 2 * i for i in range(10)}
        self.assertEqual(foo(), foo.py_func())

    def test_comprehension_basic_mixed_type(self):

        @njit
        def foo():
            return {i: float(j) for i, j in zip(range(10), range(10, 0, -1))}
        self.assertEqual(foo(), foo.py_func())

    def test_comprehension_involved(self):

        @njit
        def foo():
            a = {0: 'A', 1: 'B', 2: 'C'}
            return {3 + i: a[i] for i in range(3)}
        self.assertEqual(foo(), foo.py_func())

    def test_comprehension_fail_mixed_type(self):

        @njit
        def foo():
            a = {0: 'A', 1: 'B', 2: 1j}
            return {3 + i: a[i] for i in range(3)}
        with self.assertRaises(TypingError) as e:
            foo()
        excstr = str(e.exception)
        self.assertIn('Cannot cast complex128 to unicode_type', excstr)