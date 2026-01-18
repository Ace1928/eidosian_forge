from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
class TestIndexerBetweenTime:

    def test_indexer_between_time(self):
        rng = date_range('1/1/2000', '1/5/2000', freq='5min')
        msg = 'Cannot convert arg \\[datetime\\.datetime\\(2010, 1, 2, 1, 0\\)\\] to a time'
        with pytest.raises(ValueError, match=msg):
            rng.indexer_between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))

    @pytest.mark.parametrize('unit', ['us', 'ms', 's'])
    def test_indexer_between_time_non_nano(self, unit):
        rng = date_range('1/1/2000', '1/5/2000', freq='5min')
        arr_nano = rng._data._ndarray
        arr = arr_nano.astype(f'M8[{unit}]')
        dta = type(rng._data)._simple_new(arr, dtype=arr.dtype)
        dti = DatetimeIndex(dta)
        assert dti.dtype == arr.dtype
        tic = time(1, 25)
        toc = time(2, 29)
        result = dti.indexer_between_time(tic, toc)
        expected = rng.indexer_between_time(tic, toc)
        tm.assert_numpy_array_equal(result, expected)
        tic = time(1, 25, 0, 45678)
        toc = time(2, 29, 0, 1234)
        result = dti.indexer_between_time(tic, toc)
        expected = rng.indexer_between_time(tic, toc)
        tm.assert_numpy_array_equal(result, expected)