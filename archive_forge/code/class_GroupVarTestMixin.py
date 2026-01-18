import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
class GroupVarTestMixin:

    def test_group_var_generic_1d(self):
        prng = np.random.default_rng(2)
        out = (np.nan * np.ones((5, 1))).astype(self.dtype)
        counts = np.zeros(5, dtype='int64')
        values = 10 * prng.random((15, 1)).astype(self.dtype)
        labels = np.tile(np.arange(5), (3,)).astype('intp')
        expected_out = (np.squeeze(values).reshape((5, 3), order='F').std(axis=1, ddof=1) ** 2)[:, np.newaxis]
        expected_counts = counts + 3
        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_1d_flat_labels(self):
        prng = np.random.default_rng(2)
        out = (np.nan * np.ones((1, 1))).astype(self.dtype)
        counts = np.zeros(1, dtype='int64')
        values = 10 * prng.random((5, 1)).astype(self.dtype)
        labels = np.zeros(5, dtype='intp')
        expected_out = np.array([[values.std(ddof=1) ** 2]])
        expected_counts = counts + 5
        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_all_finite(self):
        prng = np.random.default_rng(2)
        out = (np.nan * np.ones((5, 2))).astype(self.dtype)
        counts = np.zeros(5, dtype='int64')
        values = 10 * prng.random((10, 2)).astype(self.dtype)
        labels = np.tile(np.arange(5), (2,)).astype('intp')
        expected_out = np.std(values.reshape(2, 5, 2), ddof=1, axis=0) ** 2
        expected_counts = counts + 2
        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_some_nan(self):
        prng = np.random.default_rng(2)
        out = (np.nan * np.ones((5, 2))).astype(self.dtype)
        counts = np.zeros(5, dtype='int64')
        values = 10 * prng.random((10, 2)).astype(self.dtype)
        values[:, 1] = np.nan
        labels = np.tile(np.arange(5), (2,)).astype('intp')
        expected_out = np.vstack([values[:, 0].reshape(5, 2, order='F').std(ddof=1, axis=1) ** 2, np.nan * np.ones(5)]).T.astype(self.dtype)
        expected_counts = counts + 2
        self.algo(out, counts, values, labels)
        tm.assert_almost_equal(out, expected_out, rtol=5e-07)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_constant(self):
        out = np.array([[np.nan]], dtype=self.dtype)
        counts = np.array([0], dtype='int64')
        values = 0.832845131556193 * np.ones((3, 1), dtype=self.dtype)
        labels = np.zeros(3, dtype='intp')
        self.algo(out, counts, values, labels)
        assert counts[0] == 3
        assert out[0, 0] >= 0
        tm.assert_almost_equal(out[0, 0], 0.0)