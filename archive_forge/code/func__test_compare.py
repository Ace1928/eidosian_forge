import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def _test_compare(self, pyfunc):

    def eq(pyfunc, cfunc, args):
        self.assertIs(cfunc(*args), pyfunc(*args), 'mismatch for arguments %s' % (args,))
    cfunc = jit(nopython=True)(pyfunc)
    for a, b in [((4, 5), (4, 5)), ((4, 5), (4, 6)), ((4, 6), (4, 5)), ((4, 5), (5, 4))]:
        eq(pyfunc, cfunc, (Rect(*a), Rect(*b)))
    for a, b in [((4, 5), (4, 5, 6)), ((4, 5), (4, 4, 6)), ((4, 5), (4, 6, 7))]:
        eq(pyfunc, cfunc, (Rect(*a), Point(*b)))