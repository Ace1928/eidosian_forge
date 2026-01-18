import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
class BaseTimsortTest(BaseSortingTest):

    def merge_init(self, keys):
        f = self.timsort.merge_init
        return f(keys)

    def test_binarysort(self):
        n = 20

        def check(l, n, start=0):
            res = self.array_factory(l)
            f(res, res, 0, n, start)
            self.assertSorted(l, res)
        f = self.timsort.binarysort
        l = self.sorted_list(n)
        check(l, n)
        check(l, n, n // 2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n // 2)
        check(l, n)
        check(l, n, n // 2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_binarysort_with_values(self):
        n = 20
        v = list(range(100, 100 + n))

        def check(l, n, start=0):
            res = self.array_factory(l)
            res_v = self.array_factory(v)
            f(res, res_v, 0, n, start)
            self.assertSortedValues(l, v, res, res_v)
        f = self.timsort.binarysort
        l = self.sorted_list(n)
        check(l, n)
        check(l, n, n // 2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.initially_sorted_list(n, n // 2)
        check(l, n)
        check(l, n, n // 2)
        l = self.revsorted_list(n)
        check(l, n)
        l = self.random_list(n)
        check(l, n)
        l = self.duprandom_list(n)
        check(l, n)

    def test_count_run(self):
        n = 16
        f = self.timsort.count_run

        def check(l, lo, hi):
            n, desc = f(self.array_factory(l), lo, hi)
            if desc:
                for k in range(lo, lo + n - 1):
                    a, b = (l[k], l[k + 1])
                    self.assertGreater(a, b)
                if lo + n < hi:
                    self.assertLessEqual(l[lo + n - 1], l[lo + n])
            else:
                for k in range(lo, lo + n - 1):
                    a, b = (l[k], l[k + 1])
                    self.assertLessEqual(a, b)
                if lo + n < hi:
                    self.assertGreater(l[lo + n - 1], l[lo + n], l)
        l = self.sorted_list(n, offset=100)
        check(l, 0, n)
        check(l, 1, n - 1)
        check(l, 1, 2)
        l = self.revsorted_list(n, offset=100)
        check(l, 0, n)
        check(l, 1, n - 1)
        check(l, 1, 2)
        l = self.random_list(n, offset=100)
        for i in range(len(l) - 1):
            check(l, i, n)
        l = self.duprandom_list(n, offset=100)
        for i in range(len(l) - 1):
            check(l, i, n)

    def test_gallop_left(self):
        n = 20
        f = self.timsort.gallop_left

        def check(l, key, start, stop, hint):
            k = f(key, l, start, stop, hint)
            self.assertGreaterEqual(k, start)
            self.assertLessEqual(k, stop)
            if k > start:
                self.assertLess(l[k - 1], key)
            if k < stop:
                self.assertGreaterEqual(l[k], key)

        def check_all_hints(l, key, start, stop):
            for hint in range(start, stop):
                check(l, key, start, stop, hint)

        def check_sorted_list(l):
            l = self.array_factory(l)
            for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
                check_all_hints(l, key, 0, n)
                check_all_hints(l, key, 1, n - 1)
                check_all_hints(l, key, 8, n - 8)
        l = self.sorted_list(n, offset=100)
        check_sorted_list(l)
        l = self.dupsorted_list(n, offset=100)
        check_sorted_list(l)

    def test_gallop_right(self):
        n = 20
        f = self.timsort.gallop_right

        def check(l, key, start, stop, hint):
            k = f(key, l, start, stop, hint)
            self.assertGreaterEqual(k, start)
            self.assertLessEqual(k, stop)
            if k > start:
                self.assertLessEqual(l[k - 1], key)
            if k < stop:
                self.assertGreater(l[k], key)

        def check_all_hints(l, key, start, stop):
            for hint in range(start, stop):
                check(l, key, start, stop, hint)

        def check_sorted_list(l):
            l = self.array_factory(l)
            for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
                check_all_hints(l, key, 0, n)
                check_all_hints(l, key, 1, n - 1)
                check_all_hints(l, key, 8, n - 8)
        l = self.sorted_list(n, offset=100)
        check_sorted_list(l)
        l = self.dupsorted_list(n, offset=100)
        check_sorted_list(l)

    def test_merge_compute_minrun(self):
        f = self.timsort.merge_compute_minrun
        for i in range(0, 64):
            self.assertEqual(f(i), i)
        for i in range(6, 63):
            if 2 ** i > sys.maxsize:
                break
            self.assertEqual(f(2 ** i), 32)
        for i in self.fibo():
            if i < 64:
                continue
            if i >= sys.maxsize:
                break
            k = f(i)
            self.assertGreaterEqual(k, 32)
            self.assertLessEqual(k, 64)
            if i > 500:
                quot = i // k
                p = 2 ** utils.bit_length(quot)
                self.assertLess(quot, p)
                self.assertGreaterEqual(quot, 0.9 * p)

    def check_merge_lo_hi(self, func, a, b):
        na = len(a)
        nb = len(b)
        orig_keys = [42] + a + b + [-42]
        keys = self.array_factory(orig_keys)
        ms = self.merge_init(keys)
        ssa = 1
        ssb = ssa + na
        new_ms = func(ms, keys, keys, ssa, na, ssb, nb)
        self.assertEqual(keys[0], orig_keys[0])
        self.assertEqual(keys[-1], orig_keys[-1])
        self.assertSorted(orig_keys[1:-1], keys[1:-1])
        self.assertGreaterEqual(len(new_ms.keys), len(ms.keys))
        self.assertGreaterEqual(len(new_ms.values), len(ms.values))
        self.assertIs(new_ms.pending, ms.pending)
        self.assertGreaterEqual(new_ms.min_gallop, 1)

    def test_merge_lo_hi(self):
        f_lo = self.timsort.merge_lo
        f_hi = self.timsort.merge_hi
        for na, nb in [(12, 16), (40, 40), (100, 110), (1000, 1100)]:
            for a, b in itertools.product(self.make_sample_sorted_lists(na), self.make_sample_sorted_lists(nb)):
                self.check_merge_lo_hi(f_lo, a, b)
                self.check_merge_lo_hi(f_hi, b, a)

    def check_merge_at(self, a, b):
        f = self.timsort.merge_at
        na = len(a)
        nb = len(b)
        orig_keys = [42] + a + b + [-42]
        ssa = 1
        ssb = ssa + na
        stack_sentinel = MergeRun(-42, -42)

        def run_merge_at(ms, keys, i):
            new_ms = f(ms, keys, keys, i)
            self.assertEqual(keys[0], orig_keys[0])
            self.assertEqual(keys[-1], orig_keys[-1])
            self.assertSorted(orig_keys[1:-1], keys[1:-1])
            self.assertIs(new_ms.pending, ms.pending)
            self.assertEqual(ms.pending[i], (ssa, na + nb))
            self.assertEqual(ms.pending[0], stack_sentinel)
            return new_ms
        keys = self.array_factory(orig_keys)
        ms = self.merge_init(keys)
        ms = self.timsort.merge_append(ms, stack_sentinel)
        i = ms.n
        ms = self.timsort.merge_append(ms, MergeRun(ssa, na))
        ms = self.timsort.merge_append(ms, MergeRun(ssb, nb))
        ms = run_merge_at(ms, keys, i)
        self.assertEqual(ms.n, i + 1)
        keys = self.array_factory(orig_keys)
        ms = self.merge_init(keys)
        ms = self.timsort.merge_append(ms, stack_sentinel)
        i = ms.n
        ms = self.timsort.merge_append(ms, MergeRun(ssa, na))
        ms = self.timsort.merge_append(ms, MergeRun(ssb, nb))
        last_run = MergeRun(ssb + nb, 1)
        ms = self.timsort.merge_append(ms, last_run)
        ms = run_merge_at(ms, keys, i)
        self.assertEqual(ms.n, i + 2)
        self.assertEqual(ms.pending[ms.n - 1], last_run)

    def test_merge_at(self):
        for na, nb in [(12, 16), (40, 40), (100, 110), (500, 510)]:
            for a, b in itertools.product(self.make_sample_sorted_lists(na), self.make_sample_sorted_lists(nb)):
                self.check_merge_at(a, b)
                self.check_merge_at(b, a)

    def test_merge_force_collapse(self):
        f = self.timsort.merge_force_collapse
        sizes_list = [(8, 10, 15, 20)]
        sizes_list.append(sizes_list[0][::-1])
        for sizes in sizes_list:
            for chunks in itertools.product(*(self.make_sample_sorted_lists(n) for n in sizes)):
                orig_keys = sum(chunks, [])
                keys = self.array_factory(orig_keys)
                ms = self.merge_init(keys)
                pos = 0
                for c in chunks:
                    ms = self.timsort.merge_append(ms, MergeRun(pos, len(c)))
                    pos += len(c)
                self.assertEqual(sum(ms.pending[ms.n - 1]), len(keys))
                ms = f(ms, keys, keys)
                self.assertEqual(ms.n, 1)
                self.assertEqual(ms.pending[0], MergeRun(0, len(keys)))
                self.assertSorted(orig_keys, keys)

    def test_run_timsort(self):
        f = self.timsort.run_timsort
        for size_factor in (1, 10):
            sizes = (15, 30, 20)
            all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
            for chunks in itertools.product(*all_lists):
                orig_keys = sum(chunks, [])
                keys = self.array_factory(orig_keys)
                f(keys)
                self.assertSorted(orig_keys, keys)

    def test_run_timsort_with_values(self):
        f = self.timsort.run_timsort_with_values
        for size_factor in (1, 5):
            chunk_size = 80 * size_factor
            a = self.dupsorted_list(chunk_size)
            b = self.duprandom_list(chunk_size)
            c = self.revsorted_list(chunk_size)
            orig_keys = a + b + c
            orig_values = list(range(1000, 1000 + len(orig_keys)))
            keys = self.array_factory(orig_keys)
            values = self.array_factory(orig_values)
            f(keys, values)
            self.assertSortedValues(orig_keys, orig_values, keys, values)