import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestSets(BaseTest):

    def test_constructor(self):
        pyfunc = empty_constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())
        pyfunc = constructor_usecase
        cfunc = jit(nopython=True)(pyfunc)

        def check(arg):
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    def test_set_return(self):
        pyfunc = set_return_usecase
        cfunc = jit(nopython=True)(pyfunc)
        arg = self.duplicates_array(200)
        self.assertEqual(cfunc(arg), set(arg))

    def test_iterator(self):
        pyfunc = iterator_usecase
        check = self.unordered_checker(pyfunc)
        check(self.duplicates_array(200))
        check(self.sparse_array(200))

    def test_update(self):
        pyfunc = update_usecase
        check = self.unordered_checker(pyfunc)
        a = self.sparse_array(50)
        b = self.duplicates_array(50)
        c = self.sparse_array(50)
        check(a, b, c)

    def test_remove(self):
        pyfunc = remove_usecase
        check = self.unordered_checker(pyfunc)
        a = self.sparse_array(50)
        b = a[::10]
        check(a, b)

    def test_remove_error(self):
        self.disable_leak_check()
        pyfunc = remove_usecase
        cfunc = jit(nopython=True)(pyfunc)
        items = tuple(set(self.sparse_array(3)))
        a = items[1:]
        b = (items[0],)
        with self.assertRaises(KeyError):
            cfunc(a, b)

    def test_discard(self):
        pyfunc = discard_usecase
        check = self.unordered_checker(pyfunc)
        a = self.sparse_array(50)
        b = self.sparse_array(50)
        check(a, b)

    def test_add_discard(self):
        """
        Check that the insertion logic does not create an infinite lookup
        chain with deleted entries (insertion should happen at the first
        deleted entry, not at the free entry at the end of the chain).
        See issue #1913.
        """
        pyfunc = add_discard_usecase
        check = self.unordered_checker(pyfunc)
        a = b = None
        while a == b:
            a, b = self.sparse_array(2)
        check((a,), b, b)

    def test_pop(self):
        pyfunc = pop_usecase
        check = self.unordered_checker(pyfunc)
        check(self.sparse_array(50))

    def test_contains(self):
        pyfunc = contains_usecase
        cfunc = jit(nopython=True)(pyfunc)

        def check(a, b):
            self.assertPreciseEqual(pyfunc(a, b), cfunc(a, b))
        a = self.sparse_array(50)
        b = self.sparse_array(50)
        check(a, b)

    def _test_xxx_update(self, pyfunc):
        check = self.unordered_checker(pyfunc)
        sizes = (1, 50, 500)
        for na, nb in itertools.product(sizes, sizes):
            a = self.sparse_array(na)
            b = self.sparse_array(nb)
            check(a, b)

    def test_difference_update(self):
        self._test_xxx_update(difference_update_usecase)

    def test_intersection_update(self):
        self._test_xxx_update(intersection_update_usecase)

    def test_symmetric_difference_update(self):
        self._test_xxx_update(symmetric_difference_update_usecase)

    def _test_comparator(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        def check(a, b):
            self.assertPreciseEqual(pyfunc(a, b), cfunc(a, b))
        a, b = map(set, [self.sparse_array(10), self.sparse_array(15)])
        args = [a & b, a - b, a | b, a ^ b]
        args = [tuple(x) for x in args]
        for a, b in itertools.product(args, args):
            check(a, b)

    def test_isdisjoint(self):
        self._test_comparator(isdisjoint_usecase)

    def test_issubset(self):
        self._test_comparator(issubset_usecase)

    def test_issuperset(self):
        self._test_comparator(issuperset_usecase)

    def test_clear(self):
        pyfunc = clear_usecase
        check = self.unordered_checker(pyfunc)
        check(self.sparse_array(50))

    def test_copy(self):
        pyfunc = copy_usecase
        check = self.unordered_checker(pyfunc)
        check(self.sparse_array(50))
        pyfunc = copy_usecase_empty
        check = self.unordered_checker(pyfunc)
        a = self.sparse_array(1)
        check(a)
        pyfunc = copy_usecase_deleted
        check = self.unordered_checker(pyfunc)
        check((1, 2, 4, 11), 2)
        a = self.sparse_array(50)
        check(a, a[len(a) // 2])

    def test_bool(self):
        pyfunc = bool_usecase
        check = self.unordered_checker(pyfunc)
        check(self.sparse_array(1))
        check(self.sparse_array(2))

    def _test_set_operator(self, pyfunc):
        check = self.unordered_checker(pyfunc)
        a, b = ((1, 2, 4, 11), (2, 3, 5, 11, 42))
        check(a, b)
        sizes = (1, 50, 500)
        for na, nb in itertools.product(sizes, sizes):
            a = self.sparse_array(na)
            b = self.sparse_array(nb)
            check(a, b)

    def make_operator_usecase(self, op):
        code = 'if 1:\n        def operator_usecase(a, b):\n            s = set(a) %(op)s set(b)\n            return list(s)\n        ' % dict(op=op)
        return compile_function('operator_usecase', code, globals())

    def make_inplace_operator_usecase(self, op):
        code = 'if 1:\n        def inplace_operator_usecase(a, b):\n            sa = set(a)\n            sb = set(b)\n            sc = sa\n            sc %(op)s sb\n            return list(sc), list(sa)\n        ' % dict(op=op)
        return compile_function('inplace_operator_usecase', code, globals())

    def make_comparison_usecase(self, op):
        code = 'if 1:\n        def comparison_usecase(a, b):\n            return set(a) %(op)s set(b)\n        ' % dict(op=op)
        return compile_function('comparison_usecase', code, globals())

    def test_difference(self):
        self._test_set_operator(difference_usecase)

    def test_intersection(self):
        self._test_set_operator(intersection_usecase)

    def test_symmetric_difference(self):
        self._test_set_operator(symmetric_difference_usecase)

    def test_union(self):
        self._test_set_operator(union_usecase)

    def test_and(self):
        self._test_set_operator(self.make_operator_usecase('&'))

    def test_or(self):
        self._test_set_operator(self.make_operator_usecase('|'))

    def test_sub(self):
        self._test_set_operator(self.make_operator_usecase('-'))

    def test_xor(self):
        self._test_set_operator(self.make_operator_usecase('^'))

    def test_eq(self):
        self._test_set_operator(self.make_comparison_usecase('=='))

    def test_ne(self):
        self._test_set_operator(self.make_comparison_usecase('!='))

    def test_le(self):
        self._test_set_operator(self.make_comparison_usecase('<='))

    def test_lt(self):
        self._test_set_operator(self.make_comparison_usecase('<'))

    def test_ge(self):
        self._test_set_operator(self.make_comparison_usecase('>='))

    def test_gt(self):
        self._test_set_operator(self.make_comparison_usecase('>'))

    def test_iand(self):
        self._test_set_operator(self.make_inplace_operator_usecase('&='))

    def test_ior(self):
        self._test_set_operator(self.make_inplace_operator_usecase('|='))

    def test_isub(self):
        self._test_set_operator(self.make_inplace_operator_usecase('-='))

    def test_ixor(self):
        self._test_set_operator(self.make_inplace_operator_usecase('^='))