from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
class TestLiteralLists(MemoryLeakMixin, TestCase):

    def test_basic_compile(self):

        @njit
        def foo():
            l = [1, 'a']
        foo()

    def test_literal_value_passthrough(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertTrue(isinstance(x, types.LiteralList))
            lv = x.literal_value
            self.assertTrue(isinstance(lv, list))
            self.assertEqual(lv[0], types.literal(1))
            self.assertEqual(lv[1], types.literal('a'))
            self.assertEqual(lv[2], types.Array(types.float64, 1, 'C'))
            self.assertEqual(lv[3], types.List(types.intp, reflected=False, initial_value=[1, 2, 3]))
            self.assertTrue(isinstance(lv[4], types.LiteralList))
            self.assertEqual(lv[4].literal_value[0], types.literal('cat'))
            self.assertEqual(lv[4].literal_value[1], types.literal(10))
            return lambda x: x

        @njit
        def foo():
            otherhomogeneouslist = [1, 2, 3]
            otherheterogeneouslist = ['cat', 10]
            zeros = np.zeros(5)
            l = [1, 'a', zeros, otherhomogeneouslist, otherheterogeneouslist]
            bar(l)
        foo()

    def test_literal_value_involved_passthrough(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertTrue(isinstance(x, types.LiteralStrKeyDict))
            dlv = x.literal_value
            inner_literal = {types.literal('g'): types.literal('h'), types.literal('i'): types.Array(types.float64, 1, 'C')}
            inner_dict = types.LiteralStrKeyDict(inner_literal)
            outer_literal = {types.literal('a'): types.LiteralList([types.literal(1), types.literal('a'), types.DictType(types.unicode_type, types.intp, initial_value={'f': 1}), inner_dict]), types.literal('b'): types.literal(2), types.literal('c'): types.List(types.complex128, reflected=False)}

            def check_same(a, b):
                if isinstance(a, types.LiteralList) and isinstance(b, types.LiteralList):
                    for i, j in zip(a.literal_value, b.literal_value):
                        check_same(a.literal_value, b.literal_value)
                elif isinstance(a, list) and isinstance(b, list):
                    for i, j in zip(a, b):
                        check_same(i, j)
                elif isinstance(a, types.LiteralStrKeyDict) and isinstance(b, types.LiteralStrKeyDict):
                    for (ki, vi), (kj, vj) in zip(a.literal_value.items(), b.literal_value.items()):
                        check_same(ki, kj)
                        check_same(vi, vj)
                elif isinstance(a, dict) and isinstance(b, dict):
                    for (ki, vi), (kj, vj) in zip(a.items(), b.items()):
                        check_same(ki, kj)
                        check_same(vi, vj)
                else:
                    self.assertEqual(a, b)
            check_same(dlv, outer_literal)
            return lambda x: x

        @njit
        def foo():
            l = {'a': [1, 'a', {'f': 1}, {'g': 'h', 'i': np.zeros(5)}], 'b': 2, 'c': [1j, 2j, 3j]}
            bar(l)
        foo()

    def test_mutation_failure(self):

        def staticsetitem():
            l = ['a', 1]
            l[0] = 'b'

        def delitem():
            l = ['a', 1]
            del l[0]

        def append():
            l = ['a', 1]
            l.append(2j)

        def extend():
            l = ['a', 1]
            l.extend([2j, 3j])

        def insert():
            l = ['a', 1]
            l.insert(0, 2j)

        def remove():
            l = ['a', 1]
            l.remove('a')

        def pop():
            l = ['a', 1]
            l.pop()

        def clear():
            l = ['a', 1]
            l.clear()

        def sort():
            l = ['a', 1]
            l.sort()

        def reverse():
            l = ['a', 1]
            l.reverse()
        illegals = (staticsetitem, delitem, append, extend, insert, remove, pop, clear, sort, reverse)
        for test in illegals:
            with self.subTest(test.__name__):
                with self.assertRaises(errors.TypingError) as raises:
                    njit(test)()
                expect = 'Cannot mutate a literal list'
                self.assertIn(expect, str(raises.exception))

    def test_count(self):

        @njit
        def foo():
            l = ['a', 1, 'a', 2, 'a', 3, 'b', 4, 'b', 5, 'c']
            r = []
            for x in 'abc':
                r.append(l.count(x))
            return r
        self.assertEqual(foo.py_func(), foo())

    def test_len(self):

        @njit
        def foo():
            l = ['a', 1, 'a', 2, 'a', 3, 'b', 4, 'b', 5, 'c']
            return len(l)
        self.assertEqual(foo.py_func(), foo())

    def test_contains(self):

        @njit
        def foo():
            l = ['a', 1, 'a', 2, 'a', 3, 'b', 4, 'b', 5, 'c']
            r = []
            for x in literal_unroll(('a', 'd', 2, 6)):
                r.append(x in l)
            return r
        self.assertEqual(foo.py_func(), foo())

    def test_getitem(self):

        @njit
        def foo(x):
            l = ['a', 1]
            return l[x]
        with self.assertRaises(errors.TypingError) as raises:
            foo(0)
        expect = 'Cannot __getitem__ on a literal list'
        self.assertIn(expect, str(raises.exception))

    def test_staticgetitem(self):

        @njit
        def foo():
            l = ['a', 1]
            return (l[0], l[1])
        self.assertEqual(foo.py_func(), foo())

    def test_staticgetitem_slice(self):

        @njit
        def foo():
            l = ['a', 'b', 1]
            return l[:2]
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        expect = 'Cannot __getitem__ on a literal list'
        self.assertIn(expect, str(raises.exception))

    def test_setitem(self):

        @njit
        def foo(x):
            l = ['a', 1]
            l[x] = 'b'
        with self.assertRaises(errors.TypingError) as raises:
            foo(0)
        expect = 'Cannot mutate a literal list'
        self.assertIn(expect, str(raises.exception))

    def test_unify(self):

        @njit
        def foo(x):
            if x + 1 > 3:
                l = ['a', 1]
            else:
                l = ['b', 2]
            return l[0]
        for x in (-100, 100):
            self.assertEqual(foo.py_func(x), foo(x))

    def test_not_unify(self):

        @njit
        def foo(x):
            if x + 1 > 3:
                l = ['a', 1, 2j]
            else:
                l = ['b', 2]
            return (l[0], l[1], l[0], l[1])
        with self.assertRaises(errors.TypingError) as raises:
            foo(100)
        expect = 'Cannot unify LiteralList'
        self.assertIn(expect, str(raises.exception))

    def test_index(self):

        @njit
        def foo():
            l = ['a', 1]
            l.index('a')
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        expect = 'list.index is unsupported for literal lists'
        self.assertIn(expect, str(raises.exception))

    def test_copy(self):

        @njit
        def foo():
            l = ['a', 1].copy()
            return (l[0], l[1])
        self.assertEqual(foo(), foo.py_func())

    def test_tuple_not_in_mro(self):

        def bar(x):
            pass

        @overload(bar)
        def ol_bar(x):
            self.assertFalse(isinstance(x, types.BaseTuple))
            self.assertTrue(isinstance(x, types.LiteralList))
            return lambda x: ...

        @njit
        def foo():
            l = ['a', 1]
            bar(l)
        foo()