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
def _check_element_equal(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    con = [np.arange(3).astype(np.intp), np.arange(5).astype(np.intp)]
    expect = list(con)
    pyfunc(expect)
    got = list(con)
    cfunc(got)
    self.assert_list_element_precise_equal(expect=expect, got=got)