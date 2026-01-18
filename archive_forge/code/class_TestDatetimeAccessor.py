from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
class TestDatetimeAccessor:

    @pytest.fixture(autouse=True)
    def setup(self):
        nt = 100
        data = np.random.rand(10, 10, nt)
        lons = np.linspace(0, 11, 10)
        lats = np.linspace(0, 20, 10)
        self.times = pd.date_range(start='2000/01/01', freq='h', periods=nt)
        self.data = xr.DataArray(data, coords=[lons, lats, self.times], dims=['lon', 'lat', 'time'], name='data')
        self.times_arr = np.random.choice(self.times, size=(10, 10, nt))
        self.times_data = xr.DataArray(self.times_arr, coords=[lons, lats, self.times], dims=['lon', 'lat', 'time'], name='data')

    @pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond', 'week', 'weekofyear', 'dayofweek', 'weekday', 'dayofyear', 'quarter', 'date', 'time', 'daysinmonth', 'days_in_month', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year'])
    def test_field_access(self, field) -> None:
        if field in ['week', 'weekofyear']:
            data = self.times.isocalendar()['week']
        else:
            data = getattr(self.times, field)
        if data.dtype.kind != 'b' and field not in ('date', 'time'):
            data = data.astype('int64')
        translations = {'weekday': 'dayofweek', 'daysinmonth': 'days_in_month', 'weekofyear': 'week'}
        name = translations.get(field, field)
        expected = xr.DataArray(data, name=name, coords=[self.times], dims=['time'])
        if field in ['week', 'weekofyear']:
            with pytest.warns(FutureWarning, match='dt.weekofyear and dt.week have been deprecated'):
                actual = getattr(self.data.time.dt, field)
        else:
            actual = getattr(self.data.time.dt, field)
        assert expected.dtype == actual.dtype
        assert_identical(expected, actual)

    def test_total_seconds(self) -> None:
        delta = self.data.time - np.datetime64('2000-01-03')
        actual = delta.dt.total_seconds()
        expected = xr.DataArray(np.arange(-48, 52, dtype=np.float64) * 3600, name='total_seconds', coords=[self.data.time])
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('field, pandas_field', [('year', 'year'), ('week', 'week'), ('weekday', 'day')])
    def test_isocalendar(self, field, pandas_field) -> None:
        expected = pd.Index(getattr(self.times.isocalendar(), pandas_field).astype(int))
        expected = xr.DataArray(expected, name=field, coords=[self.times], dims=['time'])
        actual = self.data.time.dt.isocalendar()[field]
        assert_equal(expected, actual)

    def test_calendar(self) -> None:
        cal = self.data.time.dt.calendar
        assert cal == 'proleptic_gregorian'

    def test_strftime(self) -> None:
        assert '2000-01-01 01:00:00' == self.data.time.dt.strftime('%Y-%m-%d %H:%M:%S')[1]

    def test_not_datetime_type(self) -> None:
        nontime_data = self.data.copy()
        int_data = np.arange(len(self.data.time)).astype('int8')
        nontime_data = nontime_data.assign_coords(time=int_data)
        with pytest.raises(AttributeError, match='dt'):
            nontime_data.time.dt

    @pytest.mark.filterwarnings('ignore:dt.weekofyear and dt.week have been deprecated')
    @requires_dask
    @pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond', 'week', 'weekofyear', 'dayofweek', 'weekday', 'dayofyear', 'quarter', 'date', 'time', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'is_leap_year'])
    def test_dask_field_access(self, field) -> None:
        import dask.array as da
        expected = getattr(self.times_data.dt, field)
        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(dask_times_arr, coords=self.data.coords, dims=self.data.dims, name='data')
        with raise_if_dask_computes():
            actual = getattr(dask_times_2d.dt, field)
        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())

    @requires_dask
    @pytest.mark.parametrize('field', ['year', 'week', 'weekday'])
    def test_isocalendar_dask(self, field) -> None:
        import dask.array as da
        expected = getattr(self.times_data.dt.isocalendar(), field)
        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(dask_times_arr, coords=self.data.coords, dims=self.data.dims, name='data')
        with raise_if_dask_computes():
            actual = dask_times_2d.dt.isocalendar()[field]
        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())

    @requires_dask
    @pytest.mark.parametrize('method, parameters', [('floor', 'D'), ('ceil', 'D'), ('round', 'D'), ('strftime', '%Y-%m-%d %H:%M:%S')])
    def test_dask_accessor_method(self, method, parameters) -> None:
        import dask.array as da
        expected = getattr(self.times_data.dt, method)(parameters)
        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(dask_times_arr, coords=self.data.coords, dims=self.data.dims, name='data')
        with raise_if_dask_computes():
            actual = getattr(dask_times_2d.dt, method)(parameters)
        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())

    def test_seasons(self) -> None:
        dates = xr.date_range(start='2000/01/01', freq='ME', periods=12, use_cftime=False)
        dates = dates.append(pd.Index([np.datetime64('NaT')]))
        dates = xr.DataArray(dates)
        seasons = xr.DataArray(['DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF', 'nan'])
        assert_array_equal(seasons.values, dates.dt.season.values)

    @pytest.mark.parametrize('method, parameters', [('floor', 'D'), ('ceil', 'D'), ('round', 'D')])
    def test_accessor_method(self, method, parameters) -> None:
        dates = pd.date_range('2014-01-01', '2014-05-01', freq='h')
        xdates = xr.DataArray(dates, dims=['time'])
        expected = getattr(dates, method)(parameters)
        actual = getattr(xdates.dt, method)(parameters)
        assert_array_equal(expected, actual)