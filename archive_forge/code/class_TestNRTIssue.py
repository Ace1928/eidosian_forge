import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
class TestNRTIssue(TestCase):

    def test_issue_with_refct_op_pruning(self):
        """
        GitHub Issue #1244 https://github.com/numba/numba/issues/1244
        """

        @njit
        def calculate_2D_vector_mag(vector):
            x, y = vector
            return math.sqrt(x ** 2 + y ** 2)

        @njit
        def normalize_2D_vector(vector):
            normalized_vector = np.empty(2, dtype=np.float64)
            mag = calculate_2D_vector_mag(vector)
            x, y = vector
            normalized_vector[0] = x / mag
            normalized_vector[1] = y / mag
            return normalized_vector

        @njit
        def normalize_vectors(num_vectors, vectors):
            normalized_vectors = np.empty((num_vectors, 2), dtype=np.float64)
            for i in range(num_vectors):
                vector = vectors[i]
                normalized_vector = normalize_2D_vector(vector)
                normalized_vectors[i, 0] = normalized_vector[0]
                normalized_vectors[i, 1] = normalized_vector[1]
            return normalized_vectors
        num_vectors = 10
        test_vectors = np.random.random((num_vectors, 2))
        got = normalize_vectors(num_vectors, test_vectors)
        expected = normalize_vectors.py_func(num_vectors, test_vectors)
        np.testing.assert_almost_equal(expected, got)

    def test_incref_after_cast(self):

        def f():
            return (0.0, np.zeros(1, dtype=np.int32))
        cfunc = njit(types.Tuple((types.complex128, types.Array(types.int32, 1, 'C')))())(f)
        z, arr = cfunc()
        self.assertPreciseEqual(z, 0j)
        self.assertPreciseEqual(arr, np.zeros(1, dtype=np.int32))

    def test_refct_pruning_issue_1511(self):

        @njit
        def f():
            a = np.ones(10, dtype=np.float64)
            b = np.ones(10, dtype=np.float64)
            return (a, b[:])
        a, b = f()
        np.testing.assert_equal(a, b)
        np.testing.assert_equal(a, np.ones(10, dtype=np.float64))

    def test_refct_pruning_issue_1526(self):

        @njit
        def udt(image, x, y):
            next_loc = np.where(image == 1)
            if len(next_loc[0]) == 0:
                y_offset = 1
                x_offset = 1
            else:
                y_offset = next_loc[0][0]
                x_offset = next_loc[1][0]
            next_loc_x = x - 1 + x_offset
            next_loc_y = y - 1 + y_offset
            return (next_loc_x, next_loc_y)
        a = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]])
        expect = udt.py_func(a, 1, 6)
        got = udt(a, 1, 6)
        self.assertEqual(expect, got)

    @TestCase.run_test_in_subprocess
    def test_no_nrt_on_njit_decoration(self):
        from numba import njit
        self.assertFalse(rtsys._init)

        @njit
        def foo():
            return 123
        self.assertFalse(rtsys._init)
        self.assertEqual(foo(), foo.py_func())
        self.assertTrue(rtsys._init)