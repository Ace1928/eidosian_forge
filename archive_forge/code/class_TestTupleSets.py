import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestTupleSets(TestSets):
    """
    Test sets with tuple keys.
    """

    def _range(self, stop):
        a = np.arange(stop, dtype=np.int64)
        b = a & 6148914691236517205
        c = (a & 2863311530).astype(np.int32)
        d = (a >> 32 & 1).astype(np.bool_)
        return list(zip(b, c, d))