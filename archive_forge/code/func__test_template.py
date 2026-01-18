from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def _test_template(self, pyfunc, argcases):
    cfunc = jit(pyfunc)
    for args in argcases:
        a1 = copy.deepcopy(args)
        a2 = copy.deepcopy(args)
        np.testing.assert_allclose(pyfunc(*a1), cfunc(*a2))