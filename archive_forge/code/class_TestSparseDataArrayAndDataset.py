from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
class TestSparseDataArrayAndDataset:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.sp_ar = sparse.random((4, 6), random_state=0, density=0.5)
        self.sp_xr = xr.DataArray(self.sp_ar, coords={'x': range(4)}, dims=('x', 'y'), name='foo')
        self.ds_ar = self.sp_ar.todense()
        self.ds_xr = xr.DataArray(self.ds_ar, coords={'x': range(4)}, dims=('x', 'y'), name='foo')

    def test_to_dataset_roundtrip(self):
        x = self.sp_xr
        assert_equal(x, x.to_dataset('x').to_dataarray('x'))

    def test_align(self):
        a1 = xr.DataArray(sparse.COO.from_numpy(np.arange(4)), dims=['x'], coords={'x': ['a', 'b', 'c', 'd']})
        b1 = xr.DataArray(sparse.COO.from_numpy(np.arange(4)), dims=['x'], coords={'x': ['a', 'b', 'd', 'e']})
        a2, b2 = xr.align(a1, b1, join='inner')
        assert isinstance(a2.data, sparse.SparseArray)
        assert isinstance(b2.data, sparse.SparseArray)
        assert np.all(a2.coords['x'].data == ['a', 'b', 'd'])
        assert np.all(b2.coords['x'].data == ['a', 'b', 'd'])

    @pytest.mark.xfail(reason='COO objects currently do not accept more than one iterable index at a time')
    def test_align_2d(self):
        A1 = xr.DataArray(self.sp_ar, dims=['x', 'y'], coords={'x': np.arange(self.sp_ar.shape[0]), 'y': np.arange(self.sp_ar.shape[1])})
        A2 = xr.DataArray(self.sp_ar, dims=['x', 'y'], coords={'x': np.arange(1, self.sp_ar.shape[0] + 1), 'y': np.arange(1, self.sp_ar.shape[1] + 1)})
        B1, B2 = xr.align(A1, A2, join='inner')
        assert np.all(B1.coords['x'] == np.arange(1, self.sp_ar.shape[0]))
        assert np.all(B1.coords['y'] == np.arange(1, self.sp_ar.shape[0]))
        assert np.all(B1.coords['x'] == B2.coords['x'])
        assert np.all(B1.coords['y'] == B2.coords['y'])

    def test_align_outer(self):
        a1 = xr.DataArray(sparse.COO.from_numpy(np.arange(4)), dims=['x'], coords={'x': ['a', 'b', 'c', 'd']})
        b1 = xr.DataArray(sparse.COO.from_numpy(np.arange(4)), dims=['x'], coords={'x': ['a', 'b', 'd', 'e']})
        a2, b2 = xr.align(a1, b1, join='outer')
        assert isinstance(a2.data, sparse.SparseArray)
        assert isinstance(b2.data, sparse.SparseArray)
        assert np.all(a2.coords['x'].data == ['a', 'b', 'c', 'd', 'e'])
        assert np.all(b2.coords['x'].data == ['a', 'b', 'c', 'd', 'e'])

    def test_concat(self):
        ds1 = xr.Dataset(data_vars={'d': self.sp_xr})
        ds2 = xr.Dataset(data_vars={'d': self.sp_xr})
        ds3 = xr.Dataset(data_vars={'d': self.sp_xr})
        out = xr.concat([ds1, ds2, ds3], dim='x')
        assert_sparse_equal(out['d'].data, sparse.concatenate([self.sp_ar, self.sp_ar, self.sp_ar], axis=0))
        out = xr.concat([self.sp_xr, self.sp_xr, self.sp_xr], dim='y')
        assert_sparse_equal(out.data, sparse.concatenate([self.sp_ar, self.sp_ar, self.sp_ar], axis=1))

    def test_stack(self):
        arr = make_xrarray({'w': 2, 'x': 3, 'y': 4})
        stacked = arr.stack(z=('x', 'y'))
        z = pd.MultiIndex.from_product([np.arange(3), np.arange(4)], names=['x', 'y'])
        expected = xr.DataArray(arr.data.reshape((2, -1)), {'w': [0, 1], 'z': z}, dims=['w', 'z'])
        assert_equal(expected, stacked)
        roundtripped = stacked.unstack()
        assert_identical(arr, roundtripped)

    def test_dataarray_repr(self):
        a = xr.DataArray(sparse.COO.from_numpy(np.ones(4)), dims=['x'], coords={'y': ('x', sparse.COO.from_numpy(np.arange(4, dtype='i8')))})
        expected = dedent('            <xarray.DataArray (x: 4)> Size: 64B\n            <COO: shape=(4,), dtype=float64, nnz=4, fill_value=0.0>\n            Coordinates:\n                y        (x) int64 48B <COO: nnz=3, fill_value=0>\n            Dimensions without coordinates: x')
        assert expected == repr(a)

    def test_dataset_repr(self):
        ds = xr.Dataset(data_vars={'a': ('x', sparse.COO.from_numpy(np.ones(4)))}, coords={'y': ('x', sparse.COO.from_numpy(np.arange(4, dtype='i8')))})
        expected = dedent('            <xarray.Dataset> Size: 112B\n            Dimensions:  (x: 4)\n            Coordinates:\n                y        (x) int64 48B <COO: nnz=3, fill_value=0>\n            Dimensions without coordinates: x\n            Data variables:\n                a        (x) float64 64B <COO: nnz=4, fill_value=0.0>')
        assert expected == repr(ds)

    @requires_dask
    def test_sparse_dask_dataset_repr(self):
        ds = xr.Dataset(data_vars={'a': ('x', sparse.COO.from_numpy(np.ones(4)))}).chunk()
        expected = dedent('            <xarray.Dataset> Size: 32B\n            Dimensions:  (x: 4)\n            Dimensions without coordinates: x\n            Data variables:\n                a        (x) float64 32B dask.array<chunksize=(4,), meta=sparse.COO>')
        assert expected == repr(ds)

    def test_dataarray_pickle(self):
        a1 = xr.DataArray(sparse.COO.from_numpy(np.ones(4)), dims=['x'], coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
        a2 = pickle.loads(pickle.dumps(a1))
        assert_identical(a1, a2)

    def test_dataset_pickle(self):
        ds1 = xr.Dataset(data_vars={'a': ('x', sparse.COO.from_numpy(np.ones(4)))}, coords={'y': ('x', sparse.COO.from_numpy(np.arange(4)))})
        ds2 = pickle.loads(pickle.dumps(ds1))
        assert_identical(ds1, ds2)

    def test_coarsen(self):
        a1 = self.ds_xr
        a2 = self.sp_xr
        m1 = a1.coarsen(x=2, boundary='trim').mean()
        m2 = a2.coarsen(x=2, boundary='trim').mean()
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason='No implementation of np.pad')
    def test_rolling(self):
        a1 = self.ds_xr
        a2 = self.sp_xr
        m1 = a1.rolling(x=2, center=True).mean()
        m2 = a2.rolling(x=2, center=True).mean()
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason='Coercion to dense')
    def test_rolling_exp(self):
        a1 = self.ds_xr
        a2 = self.sp_xr
        m1 = a1.rolling_exp(x=2, center=True).mean()
        m2 = a2.rolling_exp(x=2, center=True).mean()
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason='No implementation of np.einsum')
    def test_dot(self):
        a1 = self.xp_xr.dot(self.xp_xr[0])
        a2 = self.sp_ar.dot(self.sp_ar[0])
        assert_equal(a1, a2)

    @pytest.mark.xfail(reason='Groupby reductions produce dense output')
    def test_groupby(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        m1 = x1.groupby('x').mean(...)
        m2 = x2.groupby('x').mean(...)
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason='Groupby reductions produce dense output')
    def test_groupby_first(self):
        x = self.sp_xr.copy()
        x.coords['ab'] = ('x', ['a', 'a', 'b', 'b'])
        x.groupby('ab').first()
        x.groupby('ab').first(skipna=False)

    @pytest.mark.xfail(reason='Groupby reductions produce dense output')
    def test_groupby_bins(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        m1 = x1.groupby_bins('x', bins=[0, 3, 7, 10]).sum(...)
        m2 = x2.groupby_bins('x', bins=[0, 3, 7, 10]).sum(...)
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail(reason='Resample produces dense output')
    def test_resample(self):
        t1 = xr.DataArray(np.linspace(0, 11, num=12), coords=[pd.date_range('1999-12-15', periods=12, freq=pd.DateOffset(months=1))], dims='time')
        t2 = t1.copy()
        t2.data = sparse.COO(t2.data)
        m1 = t1.resample(time='QS-DEC').mean()
        m2 = t2.resample(time='QS-DEC').mean()
        assert isinstance(m2.data, sparse.SparseArray)
        assert np.allclose(m1.data, m2.data.todense())

    @pytest.mark.xfail
    def test_reindex(self):
        x1 = self.ds_xr
        x2 = self.sp_xr
        for kwargs in [{'x': [2, 3, 4]}, {'x': [1, 100, 2, 101, 3]}, {'x': [2.5, 3, 3.5], 'y': [2, 2.5, 3]}]:
            m1 = x1.reindex(**kwargs)
            m2 = x2.reindex(**kwargs)
            assert np.allclose(m1, m2, equal_nan=True)

    @pytest.mark.xfail
    def test_merge(self):
        x = self.sp_xr
        y = xr.merge([x, x.rename('bar')]).to_dataarray()
        assert isinstance(y, sparse.SparseArray)

    @pytest.mark.xfail
    def test_where(self):
        a = np.arange(10)
        cond = a > 3
        xr.DataArray(a).where(cond)
        s = sparse.COO.from_numpy(a)
        cond = s > 3
        xr.DataArray(s).where(cond)
        x = xr.DataArray(s)
        cond = x > 3
        x.where(cond)