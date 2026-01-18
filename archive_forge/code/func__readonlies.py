import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def _readonlies(self):
    yield b'xyz'
    yield memoryview(b'abcdefghi')
    arr = np.arange(5)
    arr.setflags(write=False)
    yield memoryview(arr)