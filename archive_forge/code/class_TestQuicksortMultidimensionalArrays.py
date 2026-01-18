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
class TestQuicksortMultidimensionalArrays(BaseSortingTest, TestCase):
    quicksort = make_jit_quicksort(is_np_array=True)
    make_quicksort = staticmethod(make_jit_quicksort)

    def assertSorted(self, orig, result):
        self.assertEqual(orig.shape, result.shape)
        self.assertPreciseEqual(orig, result)

    def array_factory(self, lst, shape=None):
        array = np.array(lst, dtype=np.float64)
        if shape is None:
            return array.reshape(-1, array.shape[0])
        else:
            return array.reshape(shape)

    def get_shapes(self, n):
        shapes = []
        if n == 1:
            return shapes
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                shapes.append((n // i, i))
                shapes.append((i, n // i))
                _shapes = self.get_shapes(n // i)
                for _shape in _shapes:
                    shapes.append((i,) + _shape)
                    shapes.append(_shape + (i,))
        return shapes

    def test_run_quicksort(self):
        f = self.quicksort.run_quicksort
        for size_factor in (1, 5):
            sizes = (15, 20)
            all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
            for chunks in itertools.product(*all_lists):
                orig_keys = sum(chunks, [])
                shape_list = self.get_shapes(len(orig_keys))
                shape_list.append(None)
                for shape in shape_list:
                    keys = self.array_factory(orig_keys, shape=shape)
                    keys_copy = self.array_factory(orig_keys, shape=shape)
                    f(keys)
                    keys_copy.sort()
                    self.assertSorted(keys_copy, keys)

    def test_run_quicksort_lt(self):

        def lt(a, b):
            return a > b
        f = self.make_quicksort(lt=lt, is_np_array=True).run_quicksort
        for size_factor in (1, 5):
            sizes = (15, 20)
            all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
            for chunks in itertools.product(*all_lists):
                orig_keys = sum(chunks, [])
                shape_list = self.get_shapes(len(orig_keys))
                shape_list.append(None)
                for shape in shape_list:
                    keys = self.array_factory(orig_keys, shape=shape)
                    keys_copy = -self.array_factory(orig_keys, shape=shape)
                    f(keys)
                    keys_copy.sort()
                    keys_copy = -keys_copy
                    self.assertSorted(keys_copy, keys)

        def lt_floats(a, b):
            return math.isnan(b) or a < b
        f = self.make_quicksort(lt=lt_floats, is_np_array=True).run_quicksort
        np.random.seed(42)
        for size in (5, 20, 50, 500):
            orig = np.random.random(size=size) * 100
            orig[np.random.random(size=size) < 0.1] = float('nan')
            orig_keys = list(orig)
            shape_list = self.get_shapes(len(orig_keys))
            shape_list.append(None)
            for shape in shape_list:
                keys = self.array_factory(orig_keys, shape=shape)
                keys_copy = self.array_factory(orig_keys, shape=shape)
                f(keys)
                keys_copy.sort()
                self.assertSorted(keys_copy, keys)