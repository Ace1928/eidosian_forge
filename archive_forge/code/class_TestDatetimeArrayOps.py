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
class TestDatetimeArrayOps(TestCase):

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

    def test_add_td(self):
        self._test_td_add_or_sub(np.add, False)

    @skip_parfors_unsupported
    def test_add_td_parallel(self):
        self._test_td_add_or_sub(np.add, True)

    def test_sub_td(self):
        self._test_td_add_or_sub(np.subtract, False)

    @skip_parfors_unsupported
    def test_sub_td_parallel(self):
        self._test_td_add_or_sub(np.subtract, True)

    def _test_add_sub_td_no_match(self, operation):
        """
        Tests that attempting to add/sub a datetime64 and timedelta64
        with types that cannot be cast raises a reasonable exception.
        """

        @njit
        def impl(a, b):
            return operation(a, b)
        fname = operation.__name__
        expected = re.escape(f"ufunc '{fname}' is not supported between datetime64[ns] and timedelta64[M]")
        with self.assertRaisesRegex((TypingError, TypeError), expected):
            impl(np.array([np.datetime64('2011-01-01')], dtype='datetime64[ns]'), np.timedelta64(1000, 'M'))

    def test_add_td_no_match(self):
        self._test_add_sub_td_no_match(np.add)

    def test_sub_td_no_match(self):
        self._test_add_sub_td_no_match(np.subtract)

    def _test_min_max(self, operation, parallel, method):
        if method:
            if operation is np.min:

                def impl(arr):
                    return arr.min()
            else:

                def impl(arr):
                    return arr.max()
        else:

            def impl(arr):
                return operation(arr)
        py_func = impl
        cfunc = njit(parallel=parallel)(impl)
        test_cases = [np.array([DT(0, 'ns'), DT(1, 'ns'), DT(2, 'ns'), DT(3, 'ns')]), np.array([DT('2011-01-01', 'ns'), DT('1971-02-02', 'ns'), DT('1900-01-01', 'ns'), DT('2021-03-03', 'ns'), DT('2004-12-07', 'ns')]), np.array([DT('2011-01-01', 'D'), DT('1971-02-02', 'D'), DT('1900-01-01', 'D'), DT('2021-03-03', 'D'), DT('2004-12-07', 'D')]), np.array([DT('2011-01-01', 'ns'), DT('1971-02-02', 'ns'), DT('1900-01-01', 'ns'), DT('2021-03-03', 'ns'), DT('2004-12-07', 'ns'), DT('NaT', 'ns')]), np.array([DT('NaT', 'ns'), DT('2011-01-01', 'ns'), DT('1971-02-02', 'ns'), DT('1900-01-01', 'ns'), DT('2021-03-03', 'ns'), DT('2004-12-07', 'ns')]), np.array([DT('1971-02-02', 'ns'), DT('NaT', 'ns')]), np.array([DT('NaT', 'ns'), DT('NaT', 'ns'), DT('NaT', 'ns')]), np.array([TD(1, 'ns'), TD(2, 'ns'), TD(3, 'ns'), TD(4, 'ns')]), np.array([TD(1, 'D'), TD(2, 'D'), TD(3, 'D'), TD(4, 'D')]), np.array([TD('NaT', 'ns'), TD(1, 'ns'), TD(2, 'ns'), TD(3, 'ns'), TD(4, 'ns')]), np.array([TD(1, 'ns'), TD(2, 'ns'), TD(3, 'ns'), TD(4, 'ns'), TD('NaT', 'ns')]), np.array([TD('NaT', 'ns')]), np.array([TD('NaT', 'ns'), TD('NaT', 'ns'), TD('NaT', 'ns')])]
        for arr in test_cases:
            py_res = py_func(arr)
            c_res = cfunc(arr)
            if np.isnat(py_res) or np.isnat(c_res):
                self.assertTrue(np.isnat(py_res))
                self.assertTrue(np.isnat(c_res))
            else:
                self.assertEqual(py_res, c_res)

    def test_min_func(self):
        self._test_min_max(min, False, False)

    def test_np_min_func(self):
        self._test_min_max(np.min, False, False)

    def test_min_method(self):
        self._test_min_max(np.min, False, True)

    def test_max_func(self):
        self._test_min_max(max, False, False)

    def test_np_max_func(self):
        self._test_min_max(np.max, False, False)

    def test_max_method(self):
        self._test_min_max(np.max, False, True)

    @skip_parfors_unsupported
    def test_min_func_parallel(self):
        self._test_min_max(np.min, True, False)

    @skip_parfors_unsupported
    def test_min_method_parallel(self):
        self._test_min_max(np.min, True, True)

    @skip_parfors_unsupported
    def test_max_func_parallel(self):
        self._test_min_max(np.max, True, False)

    @skip_parfors_unsupported
    def test_max_method_parallel(self):
        self._test_min_max(np.max, True, True)