from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
class TestReplaceSeriesCoercion(CoercionBase):
    klasses = ['series']
    method = 'replace'
    rep: dict[str, list] = {}
    rep['object'] = ['a', 'b']
    rep['int64'] = [4, 5]
    rep['float64'] = [1.1, 2.2]
    rep['complex128'] = [1 + 1j, 2 + 2j]
    rep['bool'] = [True, False]
    rep['datetime64[ns]'] = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-03')]
    for tz in ['UTC', 'US/Eastern']:
        key = f'datetime64[ns, {tz}]'
        rep[key] = [pd.Timestamp('2011-01-01', tz=tz), pd.Timestamp('2011-01-03', tz=tz)]
    rep['timedelta64[ns]'] = [pd.Timedelta('1 day'), pd.Timedelta('2 day')]

    @pytest.fixture(params=['dict', 'series'])
    def how(self, request):
        return request.param

    @pytest.fixture(params=['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]', 'timedelta64[ns]'])
    def from_key(self, request):
        return request.param

    @pytest.fixture(params=['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]', 'timedelta64[ns]'], ids=['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'datetime64tz', 'timedelta64'])
    def to_key(self, request):
        return request.param

    @pytest.fixture
    def replacer(self, how, from_key, to_key):
        """
        Object we will pass to `Series.replace`
        """
        if how == 'dict':
            replacer = dict(zip(self.rep[from_key], self.rep[to_key]))
        elif how == 'series':
            replacer = pd.Series(self.rep[to_key], index=self.rep[from_key])
        else:
            raise ValueError
        return replacer

    @pytest.mark.skipif(using_pyarrow_string_dtype(), reason='TODO: test is to complex')
    def test_replace_series(self, how, to_key, from_key, replacer):
        index = pd.Index([3, 4], name='xxx')
        obj = pd.Series(self.rep[from_key], index=index, name='yyy')
        assert obj.dtype == from_key
        if from_key.startswith('datetime') and to_key.startswith('datetime'):
            return
        elif from_key in ['datetime64[ns, US/Eastern]', 'datetime64[ns, UTC]']:
            return
        if from_key == 'float64' and to_key in 'int64' or (from_key == 'complex128' and to_key in ('int64', 'float64')):
            if not IS64 or is_platform_windows():
                pytest.skip(f'32-bit platform buggy: {from_key} -> {to_key}')
            exp = pd.Series(self.rep[to_key], index=index, name='yyy', dtype=from_key)
        else:
            exp = pd.Series(self.rep[to_key], index=index, name='yyy')
            assert exp.dtype == to_key
        msg = 'Downcasting behavior in `replace`'
        warn = FutureWarning
        if exp.dtype == obj.dtype or exp.dtype == object or (exp.dtype.kind in 'iufc' and obj.dtype.kind in 'iufc'):
            warn = None
        with tm.assert_produces_warning(warn, match=msg):
            result = obj.replace(replacer)
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize('to_key', ['timedelta64[ns]', 'bool', 'object', 'complex128', 'float64', 'int64'], indirect=True)
    @pytest.mark.parametrize('from_key', ['datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    def test_replace_series_datetime_tz(self, how, to_key, from_key, replacer, using_infer_string):
        index = pd.Index([3, 4], name='xyz')
        obj = pd.Series(self.rep[from_key], index=index, name='yyy')
        assert obj.dtype == from_key
        exp = pd.Series(self.rep[to_key], index=index, name='yyy')
        if using_infer_string and to_key == 'object':
            assert exp.dtype == 'string'
        else:
            assert exp.dtype == to_key
        msg = 'Downcasting behavior in `replace`'
        warn = FutureWarning if exp.dtype != object else None
        with tm.assert_produces_warning(warn, match=msg):
            result = obj.replace(replacer)
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize('to_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    @pytest.mark.parametrize('from_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
    def test_replace_series_datetime_datetime(self, how, to_key, from_key, replacer):
        index = pd.Index([3, 4], name='xyz')
        obj = pd.Series(self.rep[from_key], index=index, name='yyy')
        assert obj.dtype == from_key
        exp = pd.Series(self.rep[to_key], index=index, name='yyy')
        warn = FutureWarning
        if isinstance(obj.dtype, pd.DatetimeTZDtype) and isinstance(exp.dtype, pd.DatetimeTZDtype):
            exp = exp.astype(obj.dtype)
            warn = None
        else:
            assert exp.dtype == to_key
            if to_key == from_key:
                warn = None
        msg = 'Downcasting behavior in `replace`'
        with tm.assert_produces_warning(warn, match=msg):
            result = obj.replace(replacer)
        tm.assert_series_equal(result, exp)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_replace_series_period(self):
        raise NotImplementedError