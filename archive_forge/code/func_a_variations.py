import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def a_variations():
    yield [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    yield [(3, 33), (1, 111), (2, 2222)]
    yield np.full(5, fill_value=np.nan).tolist()
    yield np.linspace(-10, -5, 100).tolist()