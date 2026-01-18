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
class TestTimsortArrays(JITTimsortMixin, BaseTimsortTest, TestCase):

    def array_factory(self, lst):
        return np.array(lst, dtype=np.int32)

    def check_merge_lo_hi(self, func, a, b):
        na = len(a)
        nb = len(b)
        func = self.wrap_with_mergestate(self.timsort, func)
        orig_keys = [42] + a + b + [-42]
        keys = self.array_factory(orig_keys)
        ssa = 1
        ssb = ssa + na
        new_ms = func(keys, keys, ssa, na, ssb, nb)
        self.assertEqual(keys[0], orig_keys[0])
        self.assertEqual(keys[-1], orig_keys[-1])
        self.assertSorted(orig_keys[1:-1], keys[1:-1])