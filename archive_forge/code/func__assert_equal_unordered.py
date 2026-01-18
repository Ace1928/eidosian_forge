import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def _assert_equal_unordered(self, a, b):
    if isinstance(a, tuple):
        self.assertIsInstance(b, tuple)
        for u, v in zip(a, b):
            self._assert_equal_unordered(u, v)
    elif isinstance(a, list):
        self.assertIsInstance(b, list)
        self.assertPreciseEqual(sorted(a), sorted(b))
    else:
        self.assertPreciseEqual(a, b)