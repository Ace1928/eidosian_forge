from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
class TestDataArrayRolling:

    @pytest.mark.parametrize('da', (1, 2), indirect=True)
    @pytest.mark.parametrize('center', [True, False])
    @pytest.mark.parametrize('size', [1, 2, 3, 7])
    def test_rolling_iter(self, da: DataArray, center: bool, size: int) -> None:
        rolling_obj = da.rolling(time=size, center=center)
        rolling_obj_mean = rolling_obj.mean()
        assert len(rolling_obj.window_labels) == len(da['time'])
        assert_identical(rolling_obj.window_labels, da['time'])
        for i, (label, window_da) in enumerate(rolling_obj):
            assert label == da['time'].isel(time=i)
            actual = rolling_obj_mean.isel(time=i)
            expected = window_da.mean('time')
            np.testing.assert_allclose(actual.values, expected.values)

    @pytest.mark.parametrize('da', (1,), indirect=True)
    def test_rolling_repr(self, da) -> None:
        rolling_obj = da.rolling(time=7)
        assert repr(rolling_obj) == 'DataArrayRolling [time->7]'
        rolling_obj = da.rolling(time=7, center=True)
        assert repr(rolling_obj) == 'DataArrayRolling [time->7(center)]'
        rolling_obj = da.rolling(time=7, x=3, center=True)
        assert repr(rolling_obj) == 'DataArrayRolling [time->7(center),x->3(center)]'

    @requires_dask
    def test_repeated_rolling_rechunks(self) -> None:
        dat = DataArray(np.random.rand(7653, 300), dims=('day', 'item'))
        dat_chunk = dat.chunk({'item': 20})
        dat_chunk.rolling(day=10).mean().rolling(day=250).std()

    def test_rolling_doc(self, da) -> None:
        rolling_obj = da.rolling(time=7)
        assert '`mean`' in rolling_obj.mean.__doc__

    def test_rolling_properties(self, da) -> None:
        rolling_obj = da.rolling(time=4)
        assert rolling_obj.obj.get_axis_num('time') == 1
        with pytest.raises(ValueError, match='window must be > 0'):
            da.rolling(time=-2)
        with pytest.raises(ValueError, match='min_periods must be greater than zero'):
            da.rolling(time=2, min_periods=0)
        with pytest.raises(KeyError, match="\\('foo',\\) not found in DataArray dimensions"):
            da.rolling(foo=2)

    @pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'min', 'max', 'median', 'argmin', 'argmax'))
    @pytest.mark.parametrize('center', (True, False, None))
    @pytest.mark.parametrize('min_periods', (1, None))
    @pytest.mark.parametrize('backend', ['numpy'], indirect=True)
    def test_rolling_wrapped_bottleneck(self, da, name, center, min_periods, compute_backend) -> None:
        bn = pytest.importorskip('bottleneck', minversion='1.1')
        rolling_obj = da.rolling(time=7, min_periods=min_periods)
        func_name = f'move_{name}'
        actual = getattr(rolling_obj, name)()
        window = 7
        expected = getattr(bn, func_name)(da.values, window=window, axis=1, min_count=min_periods)
        if func_name in ['move_argmin', 'move_argmax']:
            expected = window - 1 - expected
        np.testing.assert_allclose(actual.values, expected)
        with pytest.warns(DeprecationWarning, match='Reductions are applied'):
            getattr(rolling_obj, name)(dim='time')
        rolling_obj = da.rolling(time=7, center=center)
        actual = getattr(rolling_obj, name)()['time']
        assert_allclose(actual, da['time'])

    @requires_dask
    @pytest.mark.parametrize('name', ('mean', 'count'))
    @pytest.mark.parametrize('center', (True, False, None))
    @pytest.mark.parametrize('min_periods', (1, None))
    @pytest.mark.parametrize('window', (7, 8))
    @pytest.mark.parametrize('backend', ['dask'], indirect=True)
    def test_rolling_wrapped_dask(self, da, name, center, min_periods, window) -> None:
        rolling_obj = da.rolling(time=window, min_periods=min_periods, center=center)
        actual = getattr(rolling_obj, name)().load()
        if name != 'count':
            with pytest.warns(DeprecationWarning, match='Reductions are applied'):
                getattr(rolling_obj, name)(dim='time')
        rolling_obj = da.load().rolling(time=window, min_periods=min_periods, center=center)
        expected = getattr(rolling_obj, name)()
        assert_allclose(actual, expected)
        rolling_obj = da.chunk().rolling(time=window, min_periods=min_periods, center=center)
        actual = getattr(rolling_obj, name)().load()
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('center', (True, None))
    def test_rolling_wrapped_dask_nochunk(self, center) -> None:
        pytest.importorskip('dask.array')
        da_day_clim = xr.DataArray(np.arange(1, 367), coords=[np.arange(1, 367)], dims='dayofyear')
        expected = da_day_clim.rolling(dayofyear=31, center=center).mean()
        actual = da_day_clim.chunk().rolling(dayofyear=31, center=center).mean()
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    def test_rolling_pandas_compat(self, center, window, min_periods, compute_backend) -> None:
        s = pd.Series(np.arange(10))
        da = DataArray.from_series(s)
        if min_periods is not None and window < min_periods:
            min_periods = window
        s_rolling = s.rolling(window, center=center, min_periods=min_periods).mean()
        da_rolling = da.rolling(index=window, center=center, min_periods=min_periods).mean()
        da_rolling_np = da.rolling(index=window, center=center, min_periods=min_periods).reduce(np.nanmean)
        np.testing.assert_allclose(s_rolling.values, da_rolling.values)
        np.testing.assert_allclose(s_rolling.index, da_rolling['index'])
        np.testing.assert_allclose(s_rolling.values, da_rolling_np.values)
        np.testing.assert_allclose(s_rolling.index, da_rolling_np['index'])

    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    def test_rolling_construct(self, center: bool, window: int) -> None:
        s = pd.Series(np.arange(10))
        da = DataArray.from_series(s)
        s_rolling = s.rolling(window, center=center, min_periods=1).mean()
        da_rolling = da.rolling(index=window, center=center, min_periods=1)
        da_rolling_mean = da_rolling.construct('window').mean('window')
        np.testing.assert_allclose(s_rolling.values, da_rolling_mean.values)
        np.testing.assert_allclose(s_rolling.index, da_rolling_mean['index'])
        da_rolling_mean = da_rolling.construct('window', stride=2).mean('window')
        np.testing.assert_allclose(s_rolling.values[::2], da_rolling_mean.values)
        np.testing.assert_allclose(s_rolling.index[::2], da_rolling_mean['index'])
        da_rolling_mean = da_rolling.construct('window', stride=2, fill_value=0.0).mean('window')
        assert da_rolling_mean.isnull().sum() == 0
        assert (da_rolling_mean == 0.0).sum() >= 0

    @pytest.mark.parametrize('da', (1, 2), indirect=True)
    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    @pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'max'))
    def test_rolling_reduce(self, da, center, min_periods, window, name, compute_backend) -> None:
        if min_periods is not None and window < min_periods:
            min_periods = window
        if da.isnull().sum() > 1 and window == 1:
            window = 2
        rolling_obj = da.rolling(time=window, center=center, min_periods=min_periods)
        actual = rolling_obj.reduce(getattr(np, 'nan%s' % name))
        expected = getattr(rolling_obj, name)()
        assert_allclose(actual, expected)
        assert actual.sizes == expected.sizes

    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
    @pytest.mark.parametrize('window', (1, 2, 3, 4))
    @pytest.mark.parametrize('name', ('sum', 'max'))
    def test_rolling_reduce_nonnumeric(self, center, min_periods, window, name, compute_backend) -> None:
        da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims='time').isnull()
        if min_periods is not None and window < min_periods:
            min_periods = window
        rolling_obj = da.rolling(time=window, center=center, min_periods=min_periods)
        actual = rolling_obj.reduce(getattr(np, 'nan%s' % name))
        expected = getattr(rolling_obj, name)()
        assert_allclose(actual, expected)
        assert actual.sizes == expected.sizes

    def test_rolling_count_correct(self, compute_backend) -> None:
        da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims='time')
        kwargs: list[dict[str, Any]] = [{'time': 11, 'min_periods': 1}, {'time': 11, 'min_periods': None}, {'time': 7, 'min_periods': 2}]
        expecteds = [DataArray([1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8], dims='time'), DataArray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dims='time'), DataArray([np.nan, np.nan, 2, 3, 3, 4, 5, 5, 5, 5, 5], dims='time')]
        for kwarg, expected in zip(kwargs, expecteds):
            result = da.rolling(**kwarg).count()
            assert_equal(result, expected)
            result = da.to_dataset(name='var1').rolling(**kwarg).count()['var1']
            assert_equal(result, expected)

    @pytest.mark.parametrize('da', (1,), indirect=True)
    @pytest.mark.parametrize('center', (True, False))
    @pytest.mark.parametrize('min_periods', (None, 1))
    @pytest.mark.parametrize('name', ('sum', 'mean', 'max'))
    def test_ndrolling_reduce(self, da, center, min_periods, name, compute_backend) -> None:
        rolling_obj = da.rolling(time=3, x=2, center=center, min_periods=min_periods)
        actual = getattr(rolling_obj, name)()
        expected = getattr(getattr(da.rolling(time=3, center=center, min_periods=min_periods), name)().rolling(x=2, center=center, min_periods=min_periods), name)()
        assert_allclose(actual, expected)
        assert actual.sizes == expected.sizes
        if name in ['mean']:
            expected = getattr(rolling_obj.construct({'time': 'tw', 'x': 'xw'}), name)(['tw', 'xw'])
            count = rolling_obj.count()
            if min_periods is None:
                min_periods = 1
            assert_allclose(actual, expected.where(count >= min_periods))

    @pytest.mark.parametrize('center', (True, False, (True, False)))
    @pytest.mark.parametrize('fill_value', (np.nan, 0.0))
    def test_ndrolling_construct(self, center, fill_value) -> None:
        da = DataArray(np.arange(5 * 6 * 7).reshape(5, 6, 7).astype(float), dims=['x', 'y', 'z'], coords={'x': ['a', 'b', 'c', 'd', 'e'], 'y': np.arange(6)})
        actual = da.rolling(x=3, z=2, center=center).construct(x='x1', z='z1', fill_value=fill_value)
        if not isinstance(center, tuple):
            center = (center, center)
        expected = da.rolling(x=3, center=center[0]).construct(x='x1', fill_value=fill_value).rolling(z=2, center=center[1]).construct(z='z1', fill_value=fill_value)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('funcname, argument', [('reduce', (np.mean,)), ('mean', ()), ('construct', ('window_dim',)), ('count', ())])
    def test_rolling_keep_attrs(self, funcname, argument) -> None:
        attrs_da = {'da_attr': 'test'}
        data = np.linspace(10, 15, 100)
        coords = np.linspace(1, 10, 100)
        da = DataArray(data, dims='coord', coords={'coord': coords}, attrs=attrs_da, name='name')
        func = getattr(da.rolling(dim={'coord': 5}), funcname)
        result = func(*argument)
        assert result.attrs == attrs_da
        assert result.name == 'name'
        func = getattr(da.rolling(dim={'coord': 5}), funcname)
        result = func(*argument, keep_attrs=False)
        assert result.attrs == {}
        assert result.name == 'name'
        func = getattr(da.rolling(dim={'coord': 5}), funcname)
        with set_options(keep_attrs=False):
            result = func(*argument)
        assert result.attrs == {}
        assert result.name == 'name'
        func = getattr(da.rolling(dim={'coord': 5}), funcname)
        with set_options(keep_attrs=False):
            result = func(*argument, keep_attrs=True)
        assert result.attrs == attrs_da
        assert result.name == 'name'
        func = getattr(da.rolling(dim={'coord': 5}), funcname)
        with set_options(keep_attrs=True):
            result = func(*argument, keep_attrs=False)
        assert result.attrs == {}
        assert result.name == 'name'

    @requires_dask
    @pytest.mark.parametrize('dtype', ['int', 'float32', 'float64'])
    def test_rolling_dask_dtype(self, dtype) -> None:
        data = DataArray(np.array([1, 2, 3], dtype=dtype), dims='x', coords={'x': [1, 2, 3]})
        unchunked_result = data.rolling(x=3, min_periods=1).mean()
        chunked_result = data.chunk({'x': 1}).rolling(x=3, min_periods=1).mean()
        assert chunked_result.dtype == unchunked_result.dtype