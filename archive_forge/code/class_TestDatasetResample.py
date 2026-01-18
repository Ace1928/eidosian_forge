from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
class TestDatasetResample:

    def test_resample_and_first(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        actual = ds.resample(time='1D').first(keep_attrs=True)
        expected = ds.isel(time=[0, 4, 8])
        assert_identical(expected, actual)
        expected_time = pd.date_range('2000-01-01', freq='3h', periods=19)
        expected = ds.reindex(time=expected_time)
        actual = ds.resample(time='3h')
        for how in ['mean', 'sum', 'first', 'last']:
            method = getattr(actual, how)
            result = method()
            assert_equal(expected, result)
        for method in [np.mean]:
            result = actual.reduce(method)
            assert_equal(expected, result)

    def test_resample_min_count(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        ds['foo'] = xr.where(ds['foo'] > 2.0, np.nan, ds['foo'])
        actual = ds.resample(time='1D').sum(min_count=1)
        expected = xr.concat([ds.isel(time=slice(i * 4, (i + 1) * 4)).sum('time', min_count=1) for i in range(3)], dim=actual['time'])
        assert_allclose(expected, actual)

    def test_resample_by_mean_with_keep_attrs(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        ds.attrs['dsmeta'] = 'dsdata'
        resampled_ds = ds.resample(time='1D').mean(keep_attrs=True)
        actual = resampled_ds['bar'].attrs
        expected = ds['bar'].attrs
        assert expected == actual
        actual = resampled_ds.attrs
        expected = ds.attrs
        assert expected == actual

    def test_resample_loffset(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        ds.attrs['dsmeta'] = 'dsdata'

    def test_resample_by_mean_discarding_attrs(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        ds.attrs['dsmeta'] = 'dsdata'
        resampled_ds = ds.resample(time='1D').mean(keep_attrs=False)
        assert resampled_ds['bar'].attrs == {}
        assert resampled_ds.attrs == {}

    def test_resample_by_last_discarding_attrs(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        ds.attrs['dsmeta'] = 'dsdata'
        resampled_ds = ds.resample(time='1D').last(keep_attrs=False)
        assert resampled_ds['bar'].attrs == {}
        assert resampled_ds.attrs == {}

    @requires_scipy
    def test_resample_drop_nondim_coords(self) -> None:
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range('2000-01-01', freq='6h', periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        xx, yy = np.meshgrid(xs * 5, ys * 2.5)
        tt = np.arange(len(times), dtype=int)
        array = DataArray(data, {'time': times, 'x': xs, 'y': ys}, ('x', 'y', 'time'))
        xcoord = DataArray(xx.T, {'x': xs, 'y': ys}, ('x', 'y'))
        ycoord = DataArray(yy.T, {'x': xs, 'y': ys}, ('x', 'y'))
        tcoord = DataArray(tt, {'time': times}, ('time',))
        ds = Dataset({'data': array, 'xc': xcoord, 'yc': ycoord, 'tc': tcoord})
        ds = ds.set_coords(['xc', 'yc', 'tc'])
        actual = ds.resample(time='12h').mean('time')
        assert 'tc' not in actual.coords
        actual = ds.resample(time='1h').ffill()
        assert 'tc' not in actual.coords
        actual = ds.resample(time='1h').interpolate('linear')
        assert 'tc' not in actual.coords

    def test_resample_old_api(self) -> None:
        times = pd.date_range('2000-01-01', freq='6h', periods=10)
        ds = Dataset({'foo': (['time', 'x', 'y'], np.random.randn(10, 5, 3)), 'bar': ('time', np.random.randn(10), {'meta': 'data'}), 'time': times})
        with pytest.raises(TypeError, match='resample\\(\\) no longer supports'):
            ds.resample('1D', 'time')
        with pytest.raises(TypeError, match='resample\\(\\) no longer supports'):
            ds.resample('1D', dim='time', how='mean')
        with pytest.raises(TypeError, match='resample\\(\\) no longer supports'):
            ds.resample('1D', dim='time')

    def test_resample_ds_da_are_the_same(self) -> None:
        time = pd.date_range('2000-01-01', freq='6h', periods=365 * 4)
        ds = xr.Dataset({'foo': (('time', 'x'), np.random.randn(365 * 4, 5)), 'time': time, 'x': np.arange(5)})
        assert_allclose(ds.resample(time='ME').mean()['foo'], ds.foo.resample(time='ME').mean())

    def test_ds_resample_apply_func_args(self) -> None:

        def func(arg1, arg2, arg3=0.0):
            return arg1.mean('time') + arg2 + arg3
        times = pd.date_range('2000', freq='D', periods=3)
        ds = xr.Dataset({'foo': ('time', [1.0, 1.0, 1.0]), 'time': times})
        expected = xr.Dataset({'foo': ('time', [3.0, 3.0, 3.0]), 'time': times})
        actual = ds.resample(time='D').map(func, args=(1.0,), arg3=1.0)
        assert_identical(expected, actual)