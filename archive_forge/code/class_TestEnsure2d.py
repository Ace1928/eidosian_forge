from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange
import string
import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended
class TestEnsure2d:

    @classmethod
    def setup_class(cls):
        x = np.arange(400.0).reshape((100, 4))
        cls.df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
        cls.series = cls.df.iloc[:, 0]
        cls.ndarray = x

    def test_enfore_numpy(self):
        results = tools._ensure_2d(self.df, True)
        assert_array_equal(results[0], self.ndarray)
        assert_array_equal(results[1], self.df.columns)
        results = tools._ensure_2d(self.series, True)
        assert_array_equal(results[0], self.ndarray[:, [0]])
        assert_array_equal(results[1], self.df.columns[0])

    def test_pandas(self):
        results = tools._ensure_2d(self.df, False)
        assert_frame_equal(results[0], self.df)
        assert_array_equal(results[1], self.df.columns)
        results = tools._ensure_2d(self.series, False)
        assert_frame_equal(results[0], self.df.iloc[:, [0]])
        assert_equal(results[1], self.df.columns[0])

    def test_numpy(self):
        results = tools._ensure_2d(self.ndarray)
        assert_array_equal(results[0], self.ndarray)
        assert_equal(results[1], None)
        results = tools._ensure_2d(self.ndarray[:, 0])
        assert_array_equal(results[0], self.ndarray[:, [0]])
        assert_equal(results[1], None)