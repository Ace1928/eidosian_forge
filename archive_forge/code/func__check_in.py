import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def _check_in(self, pyfunc, flags):
    dtype = types.int64
    cfunc = jit((dtype, types.UniTuple(dtype, 3)), **flags)(pyfunc)
    for i in (3, 4, 5, 6, 42):
        tup = (3, 42, 5)
        self.assertPreciseEqual(pyfunc(i, tup), cfunc(i, tup))