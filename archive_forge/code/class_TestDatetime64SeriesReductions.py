from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
class TestDatetime64SeriesReductions:

    @pytest.mark.parametrize('nat_ser', [Series([NaT, NaT]), Series([NaT, Timedelta('nat')]), Series([Timedelta('nat'), Timedelta('nat')])])
    def test_minmax_nat_series(self, nat_ser):
        assert nat_ser.min() is NaT
        assert nat_ser.max() is NaT
        assert nat_ser.min(skipna=False) is NaT
        assert nat_ser.max(skipna=False) is NaT

    @pytest.mark.parametrize('nat_df', [DataFrame([NaT, NaT]), DataFrame([NaT, Timedelta('nat')]), DataFrame([Timedelta('nat'), Timedelta('nat')])])
    def test_minmax_nat_dataframe(self, nat_df):
        assert nat_df.min()[0] is NaT
        assert nat_df.max()[0] is NaT
        assert nat_df.min(skipna=False)[0] is NaT
        assert nat_df.max(skipna=False)[0] is NaT

    def test_min_max(self):
        rng = date_range('1/1/2000', '12/31/2000')
        rng2 = rng.take(np.random.default_rng(2).permutation(len(rng)))
        the_min = rng2.min()
        the_max = rng2.max()
        assert isinstance(the_min, Timestamp)
        assert isinstance(the_max, Timestamp)
        assert the_min == rng[0]
        assert the_max == rng[-1]
        assert rng.min() == rng[0]
        assert rng.max() == rng[-1]

    def test_min_max_series(self):
        rng = date_range('1/1/2000', periods=10, freq='4h')
        lvls = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
        df = DataFrame({'TS': rng, 'V': np.random.default_rng(2).standard_normal(len(rng)), 'L': lvls})
        result = df.TS.max()
        exp = Timestamp(df.TS.iat[-1])
        assert isinstance(result, Timestamp)
        assert result == exp
        result = df.TS.min()
        exp = Timestamp(df.TS.iat[0])
        assert isinstance(result, Timestamp)
        assert result == exp