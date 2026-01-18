import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
class TestListRefctTypes(MemoryLeakMixin, TestCase):

    def test_str_item(self):

        @njit
        def foo():
            l = List.empty_list(types.unicode_type)
            for s in ('a', 'ab', 'abc', 'abcd'):
                l.append(s)
            return l
        l = foo()
        expected = ['a', 'ab', 'abc', 'abcd']
        for i, s in enumerate(expected):
            self.assertEqual(l[i], s)
        self.assertEqual(list(l), expected)
        l[3] = 'uxyz'
        self.assertEqual(l[3], 'uxyz')
        nelem = 100
        for i in range(4, nelem):
            l.append(str(i))
            self.assertEqual(l[i], str(i))

    def test_str_item_refcount_replace(self):

        @njit
        def foo():
            i, j = ('ab', 'c')
            a = i + j
            m, n = ('zy', 'x')
            z = m + n
            l = List.empty_list(types.unicode_type)
            l.append(a)
            l[0] = z
            ra, rz = (get_refcount(a), get_refcount(z))
            return (l, ra, rz)
        l, ra, rz = foo()
        self.assertEqual(l[0], 'zyx')
        self.assertEqual(ra, 1)
        self.assertEqual(rz, 2)

    def test_dict_as_item_in_list(self):

        @njit
        def foo():
            l = List.empty_list(Dict.empty(int32, int32))
            d = Dict.empty(int32, int32)
            d[0] = 1
            l.append(d)
            return get_refcount(d)
        c = foo()
        if config.LLVM_REFPRUNE_PASS:
            self.assertEqual(1, c)
        else:
            self.assertEqual(2, c)

    def test_dict_as_item_in_list_multi_refcount(self):

        @njit
        def foo():
            l = List.empty_list(Dict.empty(int32, int32))
            d = Dict.empty(int32, int32)
            d[0] = 1
            l.append(d)
            l.append(d)
            return get_refcount(d)
        c = foo()
        if config.LLVM_REFPRUNE_PASS:
            self.assertEqual(1, c)
        else:
            self.assertEqual(3, c)

    def test_list_as_value_in_dict(self):

        @njit
        def foo():
            d = Dict.empty(int32, List.empty_list(int32))
            l = List.empty_list(int32)
            l.append(0)
            d[0] = l
            return get_refcount(l)
        c = foo()
        if config.LLVM_REFPRUNE_PASS:
            self.assertEqual(1, c)
        else:
            self.assertEqual(2, c)

    def test_list_as_item_in_list(self):
        nested_type = types.ListType(types.int32)

        @njit
        def foo():
            la = List.empty_list(nested_type)
            lb = List.empty_list(types.int32)
            lb.append(1)
            la.append(lb)
            return la
        expected = foo.py_func()
        got = foo()
        self.assertEqual(expected, got)

    def test_array_as_item_in_list(self):
        nested_type = types.Array(types.float64, 1, 'C')

        @njit
        def foo():
            l = List.empty_list(nested_type)
            a = np.zeros((1,))
            l.append(a)
            return l
        expected = foo.py_func()
        got = foo()
        self.assertTrue(np.all(expected[0] == got[0]))

    def test_array_pop_from_single_value_list(self):

        @njit
        def foo():
            l = List((np.zeros((1,)),))
            l.pop()
            return l
        expected, got = (foo.py_func(), foo())
        self.assertEqual(len(expected), 0)
        self.assertEqual(len(got), 0)

    def test_5264(self):
        float_array = types.float64[:]
        l = List.empty_list(float_array)
        l.append(np.ones(3, dtype=np.float64))
        l.pop()
        self.assertEqual(0, len(l))

    def test_jitclass_as_item_in_list(self):
        spec = [('value', int32), ('array', float32[:])]

        @jitclass(spec)
        class Bag(object):

            def __init__(self, value):
                self.value = value
                self.array = np.zeros(value, dtype=np.float32)

            @property
            def size(self):
                return self.array.size

            def increment(self, val):
                for i in range(self.size):
                    self.array[i] += val
                return self.array

        @njit
        def foo():
            l = List()
            l.append(Bag(21))
            l.append(Bag(22))
            l.append(Bag(23))
            return l
        expected = foo.py_func()
        got = foo()

        def bag_equal(one, two):
            self.assertEqual(one.value, two.value)
            np.testing.assert_allclose(one.array, two.array)
        [bag_equal(a, b) for a, b in zip(expected, got)]

    def test_4960(self):

        @jitclass([('value', int32)])
        class Simple(object):

            def __init__(self, value):
                self.value = value

        @njit
        def foo():
            l = List((Simple(23), Simple(24)))
            l.pop()
            return l
        l = foo()
        self.assertEqual(1, len(l))

    def test_storage_model_mismatch(self):
        lst = List()
        ref = [('a', True, 'a'), ('b', False, 'b'), ('c', False, 'c')]
        for x in ref:
            lst.append(x)
        for i, x in enumerate(ref):
            self.assertEqual(lst[i], ref[i])

    def test_equals_on_list_with_dict_for_equal_lists(self):
        a, b = (List(), Dict())
        b['a'] = 1
        a.append(b)
        c, d = (List(), Dict())
        d['a'] = 1
        c.append(d)
        self.assertEqual(a, c)

    def test_equals_on_list_with_dict_for_unequal_dicts(self):
        a, b = (List(), Dict())
        b['a'] = 1
        a.append(b)
        c, d = (List(), Dict())
        d['a'] = 2
        c.append(d)
        self.assertNotEqual(a, c)

    def test_equals_on_list_with_dict_for_unequal_lists(self):
        a, b = (List(), Dict())
        b['a'] = 1
        a.append(b)
        c, d, e = (List(), Dict(), Dict())
        d['a'] = 1
        e['b'] = 2
        c.append(d)
        c.append(e)
        self.assertNotEqual(a, c)