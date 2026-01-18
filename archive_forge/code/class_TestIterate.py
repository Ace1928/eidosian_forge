import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
class TestIterate(unittest.TestCase):

    def test_for_loop(self):
        N = 5
        nparr = np.empty(N)
        arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
        x = 0
        for val in arr:
            x = val