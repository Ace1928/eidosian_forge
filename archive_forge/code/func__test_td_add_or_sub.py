import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def _test_td_add_or_sub(self, operation, parallel):
    """
        Test the addition/subtraction of a datetime array with a timedelta type
        """

    def impl(a, b):
        return operation(a, b)
    arr_one = np.array([np.datetime64('2011-01-01'), np.datetime64('1971-02-02'), np.datetime64('2021-03-03'), np.datetime64('2004-12-07')], dtype='datetime64[ns]')
    arr_two = np.array([np.datetime64('2011-01-01'), np.datetime64('1971-02-02'), np.datetime64('2021-03-03'), np.datetime64('2004-12-07')], dtype='datetime64[D]')
    py_func = impl
    cfunc = njit(parallel=parallel)(impl)
    test_cases = [(arr_one, np.timedelta64(1000)), (arr_two, np.timedelta64(1000)), (arr_one, np.timedelta64(-54557)), (arr_two, np.timedelta64(-54557))]
    if operation is np.add:
        test_cases.extend([(np.timedelta64(1000), arr_one), (np.timedelta64(1000), arr_two), (np.timedelta64(-54557), arr_one), (np.timedelta64(-54557), arr_two)])
    for a, b in test_cases:
        self.assertTrue(np.array_equal(py_func(a, b), cfunc(a, b)))