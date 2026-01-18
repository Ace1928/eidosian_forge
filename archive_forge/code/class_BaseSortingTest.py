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
class BaseSortingTest(object):

    def random_list(self, n, offset=10):
        random.seed(42)
        l = list(range(offset, offset + n))
        random.shuffle(l)
        return l

    def sorted_list(self, n, offset=10):
        return list(range(offset, offset + n))

    def revsorted_list(self, n, offset=10):
        return list(range(offset, offset + n))[::-1]

    def initially_sorted_list(self, n, m=None, offset=10):
        if m is None:
            m = n // 2
        l = self.sorted_list(m, offset)
        l += self.random_list(n - m, offset=l[-1] + offset)
        return l

    def duprandom_list(self, n, factor=None, offset=10):
        random.seed(42)
        if factor is None:
            factor = int(math.sqrt(n))
        l = (list(range(offset, offset + n // factor + 1)) * (factor + 1))[:n]
        assert len(l) == n
        random.shuffle(l)
        return l

    def dupsorted_list(self, n, factor=None, offset=10):
        if factor is None:
            factor = int(math.sqrt(n))
        l = (list(range(offset, offset + n // factor + 1)) * (factor + 1))[:n]
        assert len(l) == n, (len(l), n)
        l.sort()
        return l

    def assertSorted(self, orig, result):
        self.assertEqual(len(result), len(orig))
        self.assertEqual(list(result), sorted(orig))

    def assertSortedValues(self, orig, orig_values, result, result_values):
        self.assertEqual(len(result), len(orig))
        self.assertEqual(list(result), sorted(orig))
        zip_sorted = sorted(zip(orig, orig_values), key=lambda x: x[0])
        zip_result = list(zip(result, result_values))
        self.assertEqual(zip_sorted, zip_result)
        for i in range(len(zip_result) - 1):
            (k1, v1), (k2, v2) = (zip_result[i], zip_result[i + 1])
            if k1 == k2:
                self.assertLess(orig_values.index(v1), orig_values.index(v2))

    def fibo(self):
        a = 1
        b = 1
        while True:
            yield a
            a, b = (b, a + b)

    def make_sample_sorted_lists(self, n):
        lists = []
        for offset in (20, 120):
            lists.append(self.sorted_list(n, offset))
            lists.append(self.dupsorted_list(n, offset))
        return lists

    def make_sample_lists(self, n):
        lists = []
        for offset in (20, 120):
            lists.append(self.sorted_list(n, offset))
            lists.append(self.dupsorted_list(n, offset))
            lists.append(self.revsorted_list(n, offset))
            lists.append(self.duprandom_list(n, offset))
        return lists