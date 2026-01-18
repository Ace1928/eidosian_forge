import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestPeriodDtype(Base):

    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestPeriodDtype
        """
        return PeriodDtype('D')

    def test_hash_vs_equality(self, dtype):
        dtype2 = PeriodDtype('D')
        dtype3 = PeriodDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is not dtype2
        assert dtype2 is not dtype
        assert dtype3 is not dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

    def test_construction(self):
        with pytest.raises(ValueError, match='Invalid frequency: xx'):
            PeriodDtype('xx')
        for s in ['period[D]', 'Period[D]', 'D']:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day()
        for s in ['period[3D]', 'Period[3D]', '3D']:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day(3)
        for s in ['period[26h]', 'Period[26h]', '26h', 'period[1D2h]', 'Period[1D2h]', '1D2h']:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Hour(26)

    def test_cannot_use_custom_businessday(self):
        msg = 'C is not supported as period frequency'
        msg1 = '<CustomBusinessDay> is not supported as period frequency'
        msg2 = 'PeriodDtype\\[B\\] is deprecated'
        with pytest.raises(ValueError, match=msg):
            PeriodDtype('C')
        with pytest.raises(ValueError, match=msg1):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                PeriodDtype(pd.offsets.CustomBusinessDay())

    def test_subclass(self):
        a = PeriodDtype('period[D]')
        b = PeriodDtype('period[3D]')
        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_identity(self):
        assert PeriodDtype('period[D]') == PeriodDtype('period[D]')
        assert PeriodDtype('period[D]') is not PeriodDtype('period[D]')
        assert PeriodDtype('period[3D]') == PeriodDtype('period[3D]')
        assert PeriodDtype('period[3D]') is not PeriodDtype('period[3D]')
        assert PeriodDtype('period[1s1us]') == PeriodDtype('period[1000001us]')
        assert PeriodDtype('period[1s1us]') is not PeriodDtype('period[1000001us]')

    def test_compat(self, dtype):
        assert not is_datetime64_ns_dtype(dtype)
        assert not is_datetime64_ns_dtype('period[D]')
        assert not is_datetime64_dtype(dtype)
        assert not is_datetime64_dtype('period[D]')

    def test_construction_from_string(self, dtype):
        result = PeriodDtype('period[D]')
        assert is_dtype_equal(dtype, result)
        result = PeriodDtype.construct_from_string('period[D]')
        assert is_dtype_equal(dtype, result)
        with pytest.raises(TypeError, match='list'):
            PeriodDtype.construct_from_string([1, 2, 3])

    @pytest.mark.parametrize('string', ['foo', 'period[foo]', 'foo[D]', 'datetime64[ns]', 'datetime64[ns, US/Eastern]'])
    def test_construct_dtype_from_string_invalid_raises(self, string):
        msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        with pytest.raises(TypeError, match=re.escape(msg)):
            PeriodDtype.construct_from_string(string)

    def test_is_dtype(self, dtype):
        assert PeriodDtype.is_dtype(dtype)
        assert PeriodDtype.is_dtype('period[D]')
        assert PeriodDtype.is_dtype('period[3D]')
        assert PeriodDtype.is_dtype(PeriodDtype('3D'))
        assert PeriodDtype.is_dtype('period[us]')
        assert PeriodDtype.is_dtype('period[s]')
        assert PeriodDtype.is_dtype(PeriodDtype('us'))
        assert PeriodDtype.is_dtype(PeriodDtype('s'))
        assert not PeriodDtype.is_dtype('D')
        assert not PeriodDtype.is_dtype('3D')
        assert not PeriodDtype.is_dtype('U')
        assert not PeriodDtype.is_dtype('s')
        assert not PeriodDtype.is_dtype('foo')
        assert not PeriodDtype.is_dtype(np.object_)
        assert not PeriodDtype.is_dtype(np.int64)
        assert not PeriodDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        assert is_dtype_equal(dtype, 'period[D]')
        assert is_dtype_equal(dtype, PeriodDtype('D'))
        assert is_dtype_equal(dtype, PeriodDtype('D'))
        assert is_dtype_equal(PeriodDtype('D'), PeriodDtype('D'))
        assert not is_dtype_equal(dtype, 'D')
        assert not is_dtype_equal(PeriodDtype('D'), PeriodDtype('2D'))

    def test_basic(self, dtype):
        msg = 'is_period_dtype is deprecated'
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_period_dtype(dtype)
            pidx = pd.period_range('2013-01-01 09:00', periods=5, freq='h')
            assert is_period_dtype(pidx.dtype)
            assert is_period_dtype(pidx)
            s = Series(pidx, name='A')
            assert is_period_dtype(s.dtype)
            assert is_period_dtype(s)
            assert not is_period_dtype(np.dtype('float64'))
            assert not is_period_dtype(1.0)

    def test_freq_argument_required(self):
        msg = "missing 1 required positional argument: 'freq'"
        with pytest.raises(TypeError, match=msg):
            PeriodDtype()
        msg = 'PeriodDtype argument should be string or BaseOffset, got NoneType'
        with pytest.raises(TypeError, match=msg):
            PeriodDtype(None)

    def test_not_string(self):
        assert not is_string_dtype(PeriodDtype('D'))

    def test_perioddtype_caching_dateoffset_normalize(self):
        per_d = PeriodDtype(pd.offsets.YearEnd(normalize=True))
        assert per_d.freq.normalize
        per_d2 = PeriodDtype(pd.offsets.YearEnd(normalize=False))
        assert not per_d2.freq.normalize

    def test_dont_keep_ref_after_del(self):
        dtype = PeriodDtype('D')
        ref = weakref.ref(dtype)
        del dtype
        assert ref() is None