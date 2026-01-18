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
@pytest.mark.parametrize(['x', 'minindex', 'maxindex', 'nanindex'], [pytest.param(np.array([0, 1, 2, 0, -2, -4, 2]), 5, 2, None, id='int'), pytest.param(np.array([0.0, 1.0, 2.0, 0.0, -2.0, -4.0, 2.0]), 5, 2, None, id='float'), pytest.param(np.array([1.0, np.nan, 2.0, np.nan, -2.0, -4.0, 2.0]), 5, 2, 1, id='nan'), pytest.param(np.array([1.0, np.nan, 2.0, np.nan, -2.0, -4.0, 2.0]).astype('object'), 5, 2, 1, marks=pytest.mark.filterwarnings('ignore:invalid value encountered in reduce:RuntimeWarning'), id='obj'), pytest.param(np.array([np.nan, np.nan]), np.nan, np.nan, 0, id='allnan'), pytest.param(np.array(['2015-12-31', '2020-01-02', '2020-01-01', '2016-01-01'], dtype='datetime64[ns]'), 0, 1, None, id='datetime')])
class TestReduce1D(TestReduce):

    def test_min(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None) -> None:
        ar = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        if np.isnan(minindex):
            minindex = 0
        expected0 = ar.isel(x=minindex, drop=True)
        result0 = ar.min(keep_attrs=True)
        assert_identical(result0, expected0)
        result1 = ar.min()
        expected1 = expected0.copy()
        expected1.attrs = {}
        assert_identical(result1, expected1)
        result2 = ar.min(skipna=False)
        if nanindex is not None and ar.dtype.kind != 'O':
            expected2 = ar.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected1
        assert_identical(result2, expected2)

    def test_max(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None) -> None:
        ar = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        if np.isnan(minindex):
            maxindex = 0
        expected0 = ar.isel(x=maxindex, drop=True)
        result0 = ar.max(keep_attrs=True)
        assert_identical(result0, expected0)
        result1 = ar.max()
        expected1 = expected0.copy()
        expected1.attrs = {}
        assert_identical(result1, expected1)
        result2 = ar.max(skipna=False)
        if nanindex is not None and ar.dtype.kind != 'O':
            expected2 = ar.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected1
        assert_identical(result2, expected2)

    @pytest.mark.filterwarnings('ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning')
    def test_argmin(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None) -> None:
        ar = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=['x'])
        if np.isnan(minindex):
            with pytest.raises(ValueError):
                ar.argmin()
            return
        expected0 = indarr[minindex]
        result0 = ar.argmin()
        assert_identical(result0, expected0)
        result1 = ar.argmin(keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result1, expected1)
        result2 = ar.argmin(skipna=False)
        if nanindex is not None and ar.dtype.kind != 'O':
            expected2 = indarr.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected0
        assert_identical(result2, expected2)

    @pytest.mark.filterwarnings('ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning')
    def test_argmax(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None) -> None:
        ar = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=['x'])
        if np.isnan(maxindex):
            with pytest.raises(ValueError):
                ar.argmax()
            return
        expected0 = indarr[maxindex]
        result0 = ar.argmax()
        assert_identical(result0, expected0)
        result1 = ar.argmax(keep_attrs=True)
        expected1 = expected0.copy()
        expected1.attrs = self.attrs
        assert_identical(result1, expected1)
        result2 = ar.argmax(skipna=False)
        if nanindex is not None and ar.dtype.kind != 'O':
            expected2 = indarr.isel(x=nanindex, drop=True)
            expected2.attrs = {}
        else:
            expected2 = expected0
        assert_identical(result2, expected2)

    @pytest.mark.parametrize('use_dask', [True, False])
    def test_idxmin(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None, use_dask: bool) -> None:
        if use_dask and (not has_dask):
            pytest.skip('requires dask')
        if use_dask and x.dtype.kind == 'M':
            pytest.xfail("dask operation 'argmin' breaks when dtype is datetime64 (M)")
        ar0_raw = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw
        with pytest.raises(KeyError, match="'spam' not found in array dimensions"):
            ar0.idxmin(dim='spam')
        with pytest.raises(ValueError):
            xr.DataArray(5).idxmin()
        coordarr0 = xr.DataArray(ar0.coords['x'], dims=['x'])
        coordarr1 = coordarr0.copy()
        hasna = np.isnan(minindex)
        if np.isnan(minindex):
            minindex = 0
        if hasna:
            coordarr1[...] = 1
            fill_value_0 = np.nan
        else:
            fill_value_0 = 1
        expected0 = (coordarr1 * fill_value_0).isel(x=minindex, drop=True).astype('float')
        expected0.name = 'x'
        result0 = ar0.idxmin()
        assert_identical(result0, expected0)
        result1 = ar0.idxmin(fill_value=np.nan)
        assert_identical(result1, expected0)
        result2 = ar0.idxmin(keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)
        if nanindex is not None and ar0.dtype.kind != 'O':
            expected3 = coordarr0.isel(x=nanindex, drop=True).astype('float')
            expected3.name = 'x'
            expected3.attrs = {}
        else:
            expected3 = expected0.copy()
        result3 = ar0.idxmin(skipna=False)
        assert_identical(result3, expected3)
        result4 = ar0.idxmin(skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)
        if hasna:
            fill_value_5 = -1.1
        else:
            fill_value_5 = 1
        expected5 = (coordarr1 * fill_value_5).isel(x=minindex, drop=True)
        expected5.name = 'x'
        result5 = ar0.idxmin(fill_value=-1.1)
        assert_identical(result5, expected5)
        if hasna:
            fill_value_6 = -1
        else:
            fill_value_6 = 1
        expected6 = (coordarr1 * fill_value_6).isel(x=minindex, drop=True)
        expected6.name = 'x'
        result6 = ar0.idxmin(fill_value=-1)
        assert_identical(result6, expected6)
        if hasna:
            fill_value_7 = -1j
        else:
            fill_value_7 = 1
        expected7 = (coordarr1 * fill_value_7).isel(x=minindex, drop=True)
        expected7.name = 'x'
        result7 = ar0.idxmin(fill_value=-1j)
        assert_identical(result7, expected7)

    @pytest.mark.parametrize('use_dask', [True, False])
    def test_idxmax(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None, use_dask: bool) -> None:
        if use_dask and (not has_dask):
            pytest.skip('requires dask')
        if use_dask and x.dtype.kind == 'M':
            pytest.xfail("dask operation 'argmax' breaks when dtype is datetime64 (M)")
        ar0_raw = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        if use_dask:
            ar0 = ar0_raw.chunk({})
        else:
            ar0 = ar0_raw
        with pytest.raises(KeyError, match="'spam' not found in array dimensions"):
            ar0.idxmax(dim='spam')
        with pytest.raises(ValueError):
            xr.DataArray(5).idxmax()
        coordarr0 = xr.DataArray(ar0.coords['x'], dims=['x'])
        coordarr1 = coordarr0.copy()
        hasna = np.isnan(maxindex)
        if np.isnan(maxindex):
            maxindex = 0
        if hasna:
            coordarr1[...] = 1
            fill_value_0 = np.nan
        else:
            fill_value_0 = 1
        expected0 = (coordarr1 * fill_value_0).isel(x=maxindex, drop=True).astype('float')
        expected0.name = 'x'
        result0 = ar0.idxmax()
        assert_identical(result0, expected0)
        result1 = ar0.idxmax(fill_value=np.nan)
        assert_identical(result1, expected0)
        result2 = ar0.idxmax(keep_attrs=True)
        expected2 = expected0.copy()
        expected2.attrs = self.attrs
        assert_identical(result2, expected2)
        if nanindex is not None and ar0.dtype.kind != 'O':
            expected3 = coordarr0.isel(x=nanindex, drop=True).astype('float')
            expected3.name = 'x'
            expected3.attrs = {}
        else:
            expected3 = expected0.copy()
        result3 = ar0.idxmax(skipna=False)
        assert_identical(result3, expected3)
        result4 = ar0.idxmax(skipna=False, fill_value=-100j)
        assert_identical(result4, expected3)
        if hasna:
            fill_value_5 = -1.1
        else:
            fill_value_5 = 1
        expected5 = (coordarr1 * fill_value_5).isel(x=maxindex, drop=True)
        expected5.name = 'x'
        result5 = ar0.idxmax(fill_value=-1.1)
        assert_identical(result5, expected5)
        if hasna:
            fill_value_6 = -1
        else:
            fill_value_6 = 1
        expected6 = (coordarr1 * fill_value_6).isel(x=maxindex, drop=True)
        expected6.name = 'x'
        result6 = ar0.idxmax(fill_value=-1)
        assert_identical(result6, expected6)
        if hasna:
            fill_value_7 = -1j
        else:
            fill_value_7 = 1
        expected7 = (coordarr1 * fill_value_7).isel(x=maxindex, drop=True)
        expected7.name = 'x'
        result7 = ar0.idxmax(fill_value=-1j)
        assert_identical(result7, expected7)

    @pytest.mark.filterwarnings('ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning')
    def test_argmin_dim(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None) -> None:
        ar = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=['x'])
        if np.isnan(minindex):
            with pytest.raises(ValueError):
                ar.argmin()
            return
        expected0 = {'x': indarr[minindex]}
        result0 = ar.argmin(...)
        for key in expected0:
            assert_identical(result0[key], expected0[key])
        result1 = ar.argmin(..., keep_attrs=True)
        expected1 = deepcopy(expected0)
        for da in expected1.values():
            da.attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])
        result2 = ar.argmin(..., skipna=False)
        if nanindex is not None and ar.dtype.kind != 'O':
            expected2 = {'x': indarr.isel(x=nanindex, drop=True)}
            expected2['x'].attrs = {}
        else:
            expected2 = expected0
        for key in expected2:
            assert_identical(result2[key], expected2[key])

    @pytest.mark.filterwarnings('ignore:Behaviour of argmin/argmax with neither dim nor :DeprecationWarning')
    def test_argmax_dim(self, x: np.ndarray, minindex: int | float, maxindex: int | float, nanindex: int | None) -> None:
        ar = xr.DataArray(x, dims=['x'], coords={'x': np.arange(x.size) * 4}, attrs=self.attrs)
        indarr = xr.DataArray(np.arange(x.size, dtype=np.intp), dims=['x'])
        if np.isnan(maxindex):
            with pytest.raises(ValueError):
                ar.argmax()
            return
        expected0 = {'x': indarr[maxindex]}
        result0 = ar.argmax(...)
        for key in expected0:
            assert_identical(result0[key], expected0[key])
        result1 = ar.argmax(..., keep_attrs=True)
        expected1 = deepcopy(expected0)
        for da in expected1.values():
            da.attrs = self.attrs
        for key in expected1:
            assert_identical(result1[key], expected1[key])
        result2 = ar.argmax(..., skipna=False)
        if nanindex is not None and ar.dtype.kind != 'O':
            expected2 = {'x': indarr.isel(x=nanindex, drop=True)}
            expected2['x'].attrs = {}
        else:
            expected2 = expected0
        for key in expected2:
            assert_identical(result2[key], expected2[key])