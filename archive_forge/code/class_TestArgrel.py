import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
class TestArgrel:

    def test_empty(self):
        empty_array = np.array([], dtype=int)
        z1 = np.zeros(5)
        i = argrelmin(z1)
        assert_equal(len(i), 1)
        assert_array_equal(i[0], empty_array)
        z2 = np.zeros((3, 5))
        row, col = argrelmin(z2, axis=0)
        assert_array_equal(row, empty_array)
        assert_array_equal(col, empty_array)
        row, col = argrelmin(z2, axis=1)
        assert_array_equal(row, empty_array)
        assert_array_equal(col, empty_array)

    def test_basic(self):
        x = np.array([[1, 2, 2, 3, 2], [2, 1, 2, 2, 3], [3, 2, 1, 2, 2], [2, 3, 2, 1, 2], [1, 2, 3, 2, 1]])
        row, col = argrelmax(x, axis=0)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [4, 0, 1])
        row, col = argrelmax(x, axis=1)
        order = np.argsort(row)
        assert_equal(row[order], [0, 3, 4])
        assert_equal(col[order], [3, 1, 2])
        row, col = argrelmin(x, axis=0)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [1, 2, 3])
        row, col = argrelmin(x, axis=1)
        order = np.argsort(row)
        assert_equal(row[order], [1, 2, 3])
        assert_equal(col[order], [1, 2, 3])

    def test_highorder(self):
        order = 2
        sigmas = [1.0, 2.0, 10.0, 5.0, 15.0]
        test_data, act_locs = _gen_gaussians_even(sigmas, 500)
        test_data[act_locs + order] = test_data[act_locs] * 0.99999
        test_data[act_locs - order] = test_data[act_locs] * 0.99999
        rel_max_locs = argrelmax(test_data, order=order, mode='clip')[0]
        assert_(len(rel_max_locs) == len(act_locs))
        assert_((rel_max_locs == act_locs).all())

    def test_2d_gaussians(self):
        sigmas = [1.0, 2.0, 10.0]
        test_data, act_locs = _gen_gaussians_even(sigmas, 100)
        rot_factor = 20
        rot_range = np.arange(0, len(test_data)) - rot_factor
        test_data_2 = np.vstack([test_data, test_data[rot_range]])
        rel_max_rows, rel_max_cols = argrelmax(test_data_2, axis=1, order=1)
        for rw in range(0, test_data_2.shape[0]):
            inds = rel_max_rows == rw
            assert_(len(rel_max_cols[inds]) == len(act_locs))
            assert_((act_locs == rel_max_cols[inds] - rot_factor * rw).all())