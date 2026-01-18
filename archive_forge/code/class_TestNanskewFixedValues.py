from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
class TestNanskewFixedValues:

    @pytest.fixture
    def samples(self):
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_skew(self):
        return -0.1875895205961754

    @pytest.mark.parametrize('val', [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        data = val * np.ones(300)
        skew = nanops.nanskew(data)
        assert skew == 0.0

    def test_all_finite(self):
        alpha, beta = (0.3, 0.1)
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(left_tailed) < 0
        alpha, beta = (0.1, 0.3)
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(right_tailed) > 0

    def test_ground_truth(self, samples, actual_skew):
        skew = nanops.nanskew(samples)
        tm.assert_almost_equal(skew, actual_skew)

    def test_axis(self, samples, actual_skew):
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        skew = nanops.nanskew(samples, axis=1)
        tm.assert_almost_equal(skew, np.array([actual_skew, np.nan]))

    def test_nans(self, samples):
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=False)
        assert np.isnan(skew)

    def test_nans_skipna(self, samples, actual_skew):
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=True)
        tm.assert_almost_equal(skew, actual_skew)

    @property
    def prng(self):
        return np.random.default_rng(2)