import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
class TestDatetimeArrayConstructor:

    def test_from_sequence_invalid_type(self):
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        with pytest.raises(TypeError, match='Cannot create a DatetimeArray'):
            DatetimeArray._from_sequence(mi, dtype='M8[ns]')

    def test_only_1dim_accepted(self):
        arr = np.array([0, 1, 2, 3], dtype='M8[h]').astype('M8[ns]')
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match='Only 1-dimensional'):
                DatetimeArray(arr.reshape(2, 2, 1))
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match='Only 1-dimensional'):
                DatetimeArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        arr = np.arange(5, dtype=np.int64) * 3600 * 10 ** 9
        msg = 'Inferred frequency h from passed values does not conform to passed frequency W-SUN'
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, freq='W')

    @pytest.mark.parametrize('meth', [DatetimeArray._from_sequence, pd.to_datetime, pd.DatetimeIndex])
    def test_mixing_naive_tzaware_raises(self, meth):
        arr = np.array([pd.Timestamp('2000'), pd.Timestamp('2000', tz='CET')])
        msg = 'Cannot mix tz-aware with tz-naive values|Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
        for obj in [arr, arr[::-1]]:
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10 ** 9
        result = DatetimeArray._from_sequence(arr, dtype='M8[ns]')._with_freq('infer')
        expected = pd.date_range('1970-01-01', periods=5, freq='h')._data
        tm.assert_datetime_array_equal(result, expected)

    def test_mismatched_timezone_raises(self):
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            arr = DatetimeArray(np.array(['2000-01-01T06:00:00'], dtype='M8[ns]'), dtype=DatetimeTZDtype(tz='US/Central'))
        dtype = DatetimeTZDtype(tz='US/Eastern')
        msg = 'dtype=datetime64\\[ns.*\\] does not match data dtype datetime64\\[ns.*\\]'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr, dtype=dtype)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr, dtype=np.dtype('M8[ns]'))
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr.tz_localize(None), dtype=arr.dtype)

    def test_non_array_raises(self):
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match='list'):
                DatetimeArray([1, 2, 3])

    def test_bool_dtype_raises(self):
        arr = np.array([1, 2, 3], dtype='bool')
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        msg = "Unexpected value for 'dtype': 'bool'. Must be"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr)
        msg = 'dtype bool cannot be converted to datetime64\\[ns\\]'
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr, dtype='M8[ns]')
        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)
        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_incorrect_dtype_raises(self):
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype='i8'), dtype='category')
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype='i8'), dtype='m8[s]')
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype='i8'), dtype='M8[D]')

    def test_mismatched_values_dtype_units(self):
        arr = np.array([1, 2, 3], dtype='M8[s]')
        dtype = np.dtype('M8[ns]')
        msg = 'Values resolution does not match dtype.'
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, dtype=dtype)
        dtype2 = DatetimeTZDtype(tz='UTC', unit='ns')
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, dtype=dtype2)

    def test_freq_infer_raises(self):
        depr_msg = 'DatetimeArray.__init__ is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match='Frequency inference'):
                DatetimeArray(np.array([1, 2, 3], dtype='i8'), freq='infer')

    def test_copy(self):
        data = np.array([1, 2, 3], dtype='M8[ns]')
        arr = DatetimeArray._from_sequence(data, copy=False)
        assert arr._ndarray is data
        arr = DatetimeArray._from_sequence(data, copy=True)
        assert arr._ndarray is not data

    @pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
    def test_numpy_datetime_unit(self, unit):
        data = np.array([1, 2, 3], dtype=f'M8[{unit}]')
        arr = DatetimeArray._from_sequence(data)
        assert arr.unit == unit
        assert arr[0].unit == unit