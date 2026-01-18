from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@pytest.mark.parametrize(['x', 'minindex', 'maxindex', 'nanindex'], [pytest.param(np.array([[0, 1, 2, 0, -2, -4, 2], [1, 1, 1, 1, 1, 1, 1], [0, 0, -10, 5, 20, 0, 0]]), [5, 0, 2], [2, 0, 4], [None, None, None], id='int'), pytest.param(np.array([[2.0, 1.0, 2.0, 0.0, -2.0, -4.0, 2.0], [-4.0, np.nan, 2.0, np.nan, -2.0, -4.0, 2.0], [np.nan] * 7]), [5, 0, np.nan], [0, 2, np.nan], [None, 1, 0], id='nan'), pytest.param(np.array([[2.0, 1.0, 2.0, 0.0, -2.0, -4.0, 2.0], [-4.0, np.nan, 2.0, np.nan, -2.0, -4.0, 2.0], [np.nan] * 7]).astype('object'), [5, 0, np.nan], [0, 2, np.nan], [None, 1, 0], marks=pytest.mark.filterwarnings('ignore:invalid value encountered in reduce:RuntimeWarning:'), id='obj'), pytest.param(np.array([['2015-12-31', '2020-01-02', '2020-01-01', '2016-01-01'], ['2020-01-02', '2020-01-02', '2020-01-02', '2020-01-02'], ['1900-01-01', '1-02-03', '1900-01-02', '1-02-03']], dtype='datetime64[ns]'), [0, 0, 1], [1, 0, 2], [None, None, None], id='datetime')])
class TestReduce2D(TestReduce):

    def test_min(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None]) -> None:
        ar = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        minindex = [x if not np.isnan(x) else 0 for x in minindex]
        expected0list = [ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)]
        expected0 = xr.concat(expected0list, dim='y')
        result0 = ar.min(dim='x', keep_attrs=True)
        assert_identical(result0, expected0)
        result1 = ar.min(dim='x')
        expected1 = expected0
        expected1.attrs = {}
        assert_identical(result1, expected1)
        result2 = ar.min(axis=1)
        assert_identical(result2, expected1)
        minindex = [x if y is None or ar.dtype.kind == 'O' else y for x, y in zip(minindex, nanindex)]
        expected2list = [ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)]
        expected2 = xr.concat(expected2list, dim='y')
        expected2.attrs = {}
        result3 = ar.min(dim='x', skipna=False)
        assert_identical(result3, expected2)

    def test_max(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None]) -> None:
        ar = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        maxindex = [x if not np.isnan(x) else 0 for x in maxindex]
        expected0list = [ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)]
        expected0 = xr.concat(expected0list, dim='y')
        result0 = ar.max(dim='x', keep_attrs=True)
        assert_identical(result0, expected0)
        result1 = ar.max(dim='x')
        expected1 = expected0.copy()
        expected1.attrs = {}
        assert_identical(result1, expected1)
        result2 = ar.max(axis=1)
        assert_identical(result2, expected1)
        maxindex = [x if y is None or ar.dtype.kind == 'O' else y for x, y in zip(maxindex, nanindex)]
        expected2list = [ar.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)]
        expected2 = xr.concat(expected2list, dim='y')
        expected2.attrs = {}
        result3 = ar.max(dim='x', skipna=False)
        assert_identical(result3, expected2)

    def test_argmin(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None]) -> None:
        ar = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        indarrnp = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarrnp, dims=ar.dims, coords=ar.coords)
        if np.isnan(minindex).any():
            with pytest.raises(ValueError):
                ar.argmin(dim='x')
            return
        expected0list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)]
        expected0 = xr.concat(expected0list, dim='y')
        result0 = ar.argmin(dim='x')
        assert_identical(result0, expected0)
        result1 = ar.argmin(axis=1)
        assert_identical(result1, expected0)
        result2 = ar.argmin(dim='x', keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result2, expected1)
        minindex = [x if y is None or ar.dtype.kind == 'O' else y for x, y in zip(minindex, nanindex)]
        expected2list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)]
        expected2 = xr.concat(expected2list, dim='y')
        expected2.attrs = {}
        result3 = ar.argmin(dim='x', skipna=False)
        assert_identical(result3, expected2)

    def test_argmax(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None]) -> None:
        ar = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        indarr_np = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarr_np, dims=ar.dims, coords=ar.coords)
        if np.isnan(maxindex).any():
            with pytest.raises(ValueError):
                ar.argmax(dim='x')
            return
        expected0list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)]
        expected0 = xr.concat(expected0list, dim='y')
        result0 = ar.argmax(dim='x')
        assert_identical(result0, expected0)
        result1 = ar.argmax(axis=1)
        assert_identical(result1, expected0)
        result2 = ar.argmax(dim='x', keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result2, expected1)
        maxindex = [x if y is None or ar.dtype.kind == 'O' else y for x, y in zip(maxindex, nanindex)]
        expected2list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)]
        expected2 = xr.concat(expected2list, dim='y')
        expected2.attrs = {}
        result3 = ar.argmax(dim='x', skipna=False)
        assert_identical(result3, expected2)

    @pytest.mark.parametrize('use_dask', [pytest.param(True, id='dask'), pytest.param(False, id='nodask')])
    def test_idxmin(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None], use_dask: bool) -> None:
        if use_dask and (not has_dask):
            pytest.skip('requires dask')
        if use_dask and x.dtype.kind == 'M':
            pytest.xfail("dask operation 'argmin' breaks when dtype is datetime64 (M)")
        if x.dtype.kind == 'O':
            max_computes = 1
        else:
            max_computes = 0
        ar0_raw = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw
        assert_identical(ar0, ar0)
        with pytest.raises(ValueError):
            ar0.idxmin()
        with pytest.raises(KeyError):
            ar0.idxmin(dim='Y')
        assert_identical(ar0, ar0)
        coordarr0 = xr.DataArray(np.tile(ar0.coords['x'], [x.shape[0], 1]), dims=ar0.dims, coords=ar0.coords)
        hasna = [np.isnan(x) for x in minindex]
        coordarr1 = coordarr0.copy()
        coordarr1[hasna, :] = 1
        minindex0 = [x if not np.isnan(x) else 0 for x in minindex]
        nan_mult_0 = np.array([np.nan if x else 1 for x in hasna])[:, None]
        expected0list = [(coordarr1 * nan_mult_0).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex0)]
        expected0 = xr.concat(expected0list, dim='y')
        expected0.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result0 = ar0.idxmin(dim='x')
        assert_identical(result0, expected0)
        with raise_if_dask_computes(max_computes=max_computes):
            result1 = ar0.idxmin(dim='x', fill_value=np.nan)
        assert_identical(result1, expected0)
        with raise_if_dask_computes(max_computes=max_computes):
            result2 = ar0.idxmin(dim='x', keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)
        minindex3 = [x if y is None or ar0.dtype.kind == 'O' else y for x, y in zip(minindex0, nanindex)]
        expected3list = [coordarr0.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex3)]
        expected3 = xr.concat(expected3list, dim='y')
        expected3.name = 'x'
        expected3.attrs = {}
        with raise_if_dask_computes(max_computes=max_computes):
            result3 = ar0.idxmin(dim='x', skipna=False)
        assert_identical(result3, expected3)
        with raise_if_dask_computes(max_computes=max_computes):
            result4 = ar0.idxmin(dim='x', skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)
        nan_mult_5 = np.array([-1.1 if x else 1 for x in hasna])[:, None]
        expected5list = [(coordarr1 * nan_mult_5).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex0)]
        expected5 = xr.concat(expected5list, dim='y')
        expected5.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result5 = ar0.idxmin(dim='x', fill_value=-1.1)
        assert_identical(result5, expected5)
        nan_mult_6 = np.array([-1 if x else 1 for x in hasna])[:, None]
        expected6list = [(coordarr1 * nan_mult_6).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex0)]
        expected6 = xr.concat(expected6list, dim='y')
        expected6.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result6 = ar0.idxmin(dim='x', fill_value=-1)
        assert_identical(result6, expected6)
        nan_mult_7 = np.array([-5j if x else 1 for x in hasna])[:, None]
        expected7list = [(coordarr1 * nan_mult_7).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex0)]
        expected7 = xr.concat(expected7list, dim='y')
        expected7.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result7 = ar0.idxmin(dim='x', fill_value=-5j)
        assert_identical(result7, expected7)

    @pytest.mark.parametrize('use_dask', [pytest.param(True, id='dask'), pytest.param(False, id='nodask')])
    def test_idxmax(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None], use_dask: bool) -> None:
        if use_dask and (not has_dask):
            pytest.skip('requires dask')
        if use_dask and x.dtype.kind == 'M':
            pytest.xfail("dask operation 'argmax' breaks when dtype is datetime64 (M)")
        if x.dtype.kind == 'O':
            max_computes = 1
        else:
            max_computes = 0
        ar0_raw = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw
        with pytest.raises(ValueError):
            ar0.idxmax()
        with pytest.raises(KeyError):
            ar0.idxmax(dim='Y')
        ar1 = ar0.copy()
        del ar1.coords['y']
        with pytest.raises(KeyError):
            ar1.idxmax(dim='y')
        coordarr0 = xr.DataArray(np.tile(ar0.coords['x'], [x.shape[0], 1]), dims=ar0.dims, coords=ar0.coords)
        hasna = [np.isnan(x) for x in maxindex]
        coordarr1 = coordarr0.copy()
        coordarr1[hasna, :] = 1
        maxindex0 = [x if not np.isnan(x) else 0 for x in maxindex]
        nan_mult_0 = np.array([np.nan if x else 1 for x in hasna])[:, None]
        expected0list = [(coordarr1 * nan_mult_0).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex0)]
        expected0 = xr.concat(expected0list, dim='y')
        expected0.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result0 = ar0.idxmax(dim='x')
        assert_identical(result0, expected0)
        with raise_if_dask_computes(max_computes=max_computes):
            result1 = ar0.idxmax(dim='x', fill_value=np.nan)
        assert_identical(result1, expected0)
        with raise_if_dask_computes(max_computes=max_computes):
            result2 = ar0.idxmax(dim='x', keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)
        maxindex3 = [x if y is None or ar0.dtype.kind == 'O' else y for x, y in zip(maxindex0, nanindex)]
        expected3list = [coordarr0.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex3)]
        expected3 = xr.concat(expected3list, dim='y')
        expected3.name = 'x'
        expected3.attrs = {}
        with raise_if_dask_computes(max_computes=max_computes):
            result3 = ar0.idxmax(dim='x', skipna=False)
        assert_identical(result3, expected3)
        with raise_if_dask_computes(max_computes=max_computes):
            result4 = ar0.idxmax(dim='x', skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)
        nan_mult_5 = np.array([-1.1 if x else 1 for x in hasna])[:, None]
        expected5list = [(coordarr1 * nan_mult_5).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex0)]
        expected5 = xr.concat(expected5list, dim='y')
        expected5.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result5 = ar0.idxmax(dim='x', fill_value=-1.1)
        assert_identical(result5, expected5)
        nan_mult_6 = np.array([-1 if x else 1 for x in hasna])[:, None]
        expected6list = [(coordarr1 * nan_mult_6).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex0)]
        expected6 = xr.concat(expected6list, dim='y')
        expected6.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result6 = ar0.idxmax(dim='x', fill_value=-1)
        assert_identical(result6, expected6)
        nan_mult_7 = np.array([-5j if x else 1 for x in hasna])[:, None]
        expected7list = [(coordarr1 * nan_mult_7).isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex0)]
        expected7 = xr.concat(expected7list, dim='y')
        expected7.name = 'x'
        with raise_if_dask_computes(max_computes=max_computes):
            result7 = ar0.idxmax(dim='x', fill_value=-5j)
        assert_identical(result7, expected7)

    @pytest.mark.filterwarnings('ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning')
    def test_argmin_dim(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None]) -> None:
        ar = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        indarrnp = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarrnp, dims=ar.dims, coords=ar.coords)
        if np.isnan(minindex).any():
            with pytest.raises(ValueError):
                ar.argmin(dim='x')
            return
        expected0list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)]
        expected0 = {'x': xr.concat(expected0list, dim='y')}
        result0 = ar.argmin(dim=['x'])
        for key in expected0:
            assert_identical(result0[key], expected0[key])
        result1 = ar.argmin(dim=['x'], keep_attrs=True)
        expected1 = deepcopy(expected0)
        expected1['x'].attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])
        minindex = [x if y is None or ar.dtype.kind == 'O' else y for x, y in zip(minindex, nanindex)]
        expected2list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(minindex)]
        expected2 = {'x': xr.concat(expected2list, dim='y')}
        expected2['x'].attrs = {}
        result2 = ar.argmin(dim=['x'], skipna=False)
        for key in expected2:
            assert_identical(result2[key], expected2[key])
        result3 = ar.argmin(...)
        min_xind = cast(DataArray, ar.isel(expected0).argmin())
        expected3 = {'y': DataArray(min_xind), 'x': DataArray(minindex[min_xind.item()])}
        for key in expected3:
            assert_identical(result3[key], expected3[key])

    @pytest.mark.filterwarnings('ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning')
    def test_argmax_dim(self, x: np.ndarray, minindex: list[int | float], maxindex: list[int | float], nanindex: list[int | None]) -> None:
        ar = xr.DataArray(x, dims=['y', 'x'], coords={'x': np.arange(x.shape[1]) * 4, 'y': 1 - np.arange(x.shape[0])}, attrs=self.attrs)
        indarrnp = np.tile(np.arange(x.shape[1], dtype=np.intp), [x.shape[0], 1])
        indarr = xr.DataArray(indarrnp, dims=ar.dims, coords=ar.coords)
        if np.isnan(maxindex).any():
            with pytest.raises(ValueError):
                ar.argmax(dim='x')
            return
        expected0list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)]
        expected0 = {'x': xr.concat(expected0list, dim='y')}
        result0 = ar.argmax(dim=['x'])
        for key in expected0:
            assert_identical(result0[key], expected0[key])
        result1 = ar.argmax(dim=['x'], keep_attrs=True)
        expected1 = deepcopy(expected0)
        expected1['x'].attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])
        maxindex = [x if y is None or ar.dtype.kind == 'O' else y for x, y in zip(maxindex, nanindex)]
        expected2list = [indarr.isel(y=yi).isel(x=indi, drop=True) for yi, indi in enumerate(maxindex)]
        expected2 = {'x': xr.concat(expected2list, dim='y')}
        expected2['x'].attrs = {}
        result2 = ar.argmax(dim=['x'], skipna=False)
        for key in expected2:
            assert_identical(result2[key], expected2[key])
        result3 = ar.argmax(...)
        max_xind = cast(DataArray, ar.isel(expected0).argmax())
        expected3 = {'y': DataArray(max_xind), 'x': DataArray(maxindex[max_xind.item()])}
        for key in expected3:
            assert_identical(result3[key], expected3[key])