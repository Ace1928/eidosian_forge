from datetime import timedelta
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSeriesMissingData:

    def test_categorical_nan_handling(self):
        s = Series(Categorical(['a', 'b', np.nan, 'a']))
        tm.assert_index_equal(s.cat.categories, Index(['a', 'b']))
        tm.assert_numpy_array_equal(s.values.codes, np.array([0, 1, -1, 0], dtype=np.int8))

    def test_isna_for_inf(self):
        s = Series(['a', np.inf, np.nan, pd.NA, 1.0])
        msg = 'use_inf_as_na option is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context('mode.use_inf_as_na', True):
                r = s.isna()
                dr = s.dropna()
        e = Series([False, True, True, True, False])
        de = Series(['a', 1.0], index=[0, 4])
        tm.assert_series_equal(r, e)
        tm.assert_series_equal(dr, de)

    def test_timedelta64_nan(self):
        td = Series([timedelta(days=i) for i in range(10)])
        td1 = td.copy()
        td1[0] = np.nan
        assert isna(td1[0])
        assert td1[0]._value == iNaT
        td1[0] = td[0]
        assert not isna(td1[0])
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            td1[1] = iNaT
        assert not isna(td1[1])
        assert td1.dtype == np.object_
        assert td1[1] == iNaT
        td1[1] = td[1]
        assert not isna(td1[1])
        td1[2] = NaT
        assert isna(td1[2])
        assert td1[2]._value == iNaT
        td1[2] = td[2]
        assert not isna(td1[2])
        td3 = np.timedelta64(timedelta(days=3))
        td7 = np.timedelta64(timedelta(days=7))
        td[(td > td3) & (td < td7)] = np.nan
        assert isna(td).sum() == 3

    @pytest.mark.xfail(reason="Chained inequality raises when trying to define 'selector'")
    def test_logical_range_select(self, datetime_series):
        selector = -0.5 <= datetime_series <= 0.5
        expected = (datetime_series >= -0.5) & (datetime_series <= 0.5)
        tm.assert_series_equal(selector, expected)

    def test_valid(self, datetime_series):
        ts = datetime_series.copy()
        ts.index = ts.index._with_freq(None)
        ts[::2] = np.nan
        result = ts.dropna()
        assert len(result) == ts.count()
        tm.assert_series_equal(result, ts[1::2])
        tm.assert_series_equal(result, ts[pd.notna(ts)])