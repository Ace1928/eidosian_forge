from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def _shape_variations(n):
    yield (n, n)
    yield (2 * n, n)
    yield (n, 2 * n)
    yield (2 * n + 1, 2 * n - 1)
    yield (n, n, n, n)
    yield (1, 1, 1)