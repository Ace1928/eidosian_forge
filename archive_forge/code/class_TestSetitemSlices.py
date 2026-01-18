from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
class TestSetitemSlices:

    def test_setitem_slice_float_raises(self, datetime_series):
        msg = 'cannot do slice indexing on DatetimeIndex with these indexers \\[{key}\\] of type float'
        with pytest.raises(TypeError, match=msg.format(key='4\\.0')):
            datetime_series[4.0:10.0] = 0
        with pytest.raises(TypeError, match=msg.format(key='4\\.5')):
            datetime_series[4.5:10.0] = 0

    def test_setitem_slice(self):
        ser = Series(range(10), index=list(range(10)))
        ser[-12:] = 0
        assert (ser == 0).all()
        ser[:-12] = 5
        assert (ser == 0).all()

    def test_setitem_slice_integers(self):
        ser = Series(np.random.default_rng(2).standard_normal(8), index=[2, 4, 6, 8, 10, 12, 14, 16])
        ser[:4] = 0
        assert (ser[:4] == 0).all()
        assert not (ser[4:] == 0).any()

    def test_setitem_slicestep(self):
        series = Series(np.arange(20, dtype=np.float64), index=np.arange(20, dtype=np.int64))
        series[::2] = 0
        assert (series[::2] == 0).all()

    def test_setitem_multiindex_slice(self, indexer_sli):
        mi = MultiIndex.from_product(([0, 1], list('abcde')))
        result = Series(np.arange(10, dtype=np.int64), mi)
        indexer_sli(result)[::4] = 100
        expected = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
        tm.assert_series_equal(result, expected)