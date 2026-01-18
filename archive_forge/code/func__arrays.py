import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def _arrays(self):
    arr = np.arange(12)
    yield arr
    arr = arr.reshape((3, 4))
    yield arr
    yield arr.T
    yield arr[::2]
    arr.setflags(write=False)
    yield arr
    arr = np.zeros(())
    assert arr.ndim == 0
    yield arr