import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def _test_set_equal(self, pyfunc, value, valuetype):
    rec = numpy_support.from_dtype(recordtype)
    cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp, valuetype))
    for i in range(self.sample1d.size):
        got = self.sample1d.copy()
        expect = got.copy().view(np.recarray)
        cfunc[1, 1](got, i, value)
        pyfunc(expect, i, value)
        self.assertTrue(np.all(expect == got))