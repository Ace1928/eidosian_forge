import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def _run_parallel(self, func, *args, **kwargs):
    cfunc = njit(parallel=True)(func)
    expect = func(*args, **kwargs)
    got = cfunc(*args, **kwargs)
    return (expect, got)