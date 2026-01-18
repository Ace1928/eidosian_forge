from __future__ import print_function, absolute_import, division
import unittest
import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase
def _run_test_for_gufunc(self, gufunc, py_func, expect_f4_to_pass=True, z=2):
    for dtype, expect_to_pass in [('f8', True), ('f4', expect_f4_to_pass)]:
        inputs = [np.zeros(10, dtype) for _ in range(gufunc.nin - 1)]
        ex_inputs = [x_t.copy() for x_t in inputs]
        gufunc(*inputs, z)
        py_func(*ex_inputs, np.array([z]))
        for i, (x_t, ex_x_t) in enumerate(zip(inputs, ex_inputs)):
            if expect_to_pass:
                np.testing.assert_equal(x_t, ex_x_t, err_msg='input %s' % i)
            else:
                self.assertFalse((x_t == ex_x_t).all(), msg='input %s' % i)