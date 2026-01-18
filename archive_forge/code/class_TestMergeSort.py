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
class TestMergeSort(TestCase):

    def setUp(self):
        np.random.seed(321)

    def check_argsort_stable(self, sorter, low, high, count):
        data = np.random.randint(low, high, count)
        expect = np.argsort(data, kind='mergesort')
        got = sorter(data)
        np.testing.assert_equal(expect, got)

    def test_argsort_stable(self):
        arglist = [(-2, 2, 5), (-5, 5, 10), (0, 10, 101), (0, 100, 1003)]
        imp = make_jit_mergesort(is_argsort=True)
        toplevel = imp.run_mergesort
        sorter = njit(lambda arr: toplevel(arr))
        for args in arglist:
            self.check_argsort_stable(sorter, *args)