import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def _test_comparator(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)

    def check(a, b):
        self.assertPreciseEqual(pyfunc(a, b), cfunc(a, b))
    a, b = map(set, [self.sparse_array(10), self.sparse_array(15)])
    args = [a & b, a - b, a | b, a ^ b]
    args = [tuple(x) for x in args]
    for a, b in itertools.product(args, args):
        check(a, b)