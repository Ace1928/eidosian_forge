from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def _typing_error(msg, *args):
    with self.assertRaises(errors.TypingError) as raises:
        sliding_window_view(*args)
    self.assertIn(msg, str(raises.exception))