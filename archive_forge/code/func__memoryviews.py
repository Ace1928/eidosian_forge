import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def _memoryviews(self):
    n = 10
    yield memoryview(bytearray(b'abcdefghi'))
    yield memoryview(b'abcdefghi')
    for dtype, start, stop in [('int8', -10, 10), ('uint8', 0, 10), ('int16', -5000, 1000), ('uint16', 40000, 50000), ('int32', -100000, 100000), ('uint32', 0, 1000000), ('int64', -2 ** 60, 10), ('uint64', 0, 2 ** 64 - 10), ('float32', 1.5, 3.5), ('float64', 1.5, 3.5), ('complex64', -8j, 12 + 5j), ('complex128', -8j, 12 + 5j)]:
        yield memoryview(np.linspace(start, stop, n).astype(dtype))
    arr = np.arange(12).reshape((3, 4))
    assert arr.flags.c_contiguous and (not arr.flags.f_contiguous)
    yield memoryview(arr)
    arr = arr.T
    assert arr.flags.f_contiguous and (not arr.flags.c_contiguous)
    yield memoryview(arr)
    arr = arr[::2]
    assert not arr.flags.f_contiguous and (not arr.flags.c_contiguous)
    yield memoryview(arr)