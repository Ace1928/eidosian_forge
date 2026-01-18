from __future__ import annotations
import contextlib
import gzip
import itertools
import math
import os.path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from io import BytesIO
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.errors import OutOfBoundsDatetime
import xarray as xr
from xarray import (
from xarray.backends.common import robust_getitem
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.backends.netCDF4_ import (
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.scipy_ import ScipyBackendEntrypoint
from xarray.coding.cftime_offsets import cftime_range
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_coding_times import (
from xarray.tests.test_dataset import (
@requires_dask
@requires_scipy
@requires_netCDF4
class TestDask(DatasetIOBase):

    @contextlib.contextmanager
    def create_store(self):
        yield Dataset()

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False):
        yield data.chunk()

    def test_roundtrip_string_encoded_characters(self) -> None:
        pass

    def test_roundtrip_coordinates_with_space(self) -> None:
        pass

    def test_roundtrip_numpy_datetime_data(self) -> None:
        times = pd.to_datetime(['2000-01-01', '2000-01-02', 'NaT'])
        expected = Dataset({'t': ('t', times), 't0': times[0]})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_cftime_datetime_data(self) -> None:
        from xarray.tests.test_coding_times import _all_cftime_date_types
        date_types = _all_cftime_date_types()
        for date_type in date_types.values():
            times = [date_type(1, 1, 1), date_type(1, 1, 2)]
            expected = Dataset({'t': ('t', times), 't0': times[0]})
            expected_decoded_t = np.array(times)
            expected_decoded_t0 = np.array([date_type(1, 1, 1)])
            with self.roundtrip(expected) as actual:
                abs_diff = abs(actual.t.values - expected_decoded_t)
                assert (abs_diff <= np.timedelta64(1, 's')).all()
                abs_diff = abs(actual.t0.values - expected_decoded_t0)
                assert (abs_diff <= np.timedelta64(1, 's')).all()

    def test_write_store(self) -> None:
        pass

    def test_dataset_caching(self) -> None:
        expected = Dataset({'foo': ('x', [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert not actual.foo.variable._in_memory
            actual.foo.values
            assert not actual.foo.variable._in_memory

    def test_open_mfdataset(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested') as actual:
                    assert isinstance(actual.foo.variable.data, da.Array)
                    assert actual.foo.variable.data.chunks == ((5, 5),)
                    assert_identical(original, actual)
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested', chunks={'x': 3}) as actual:
                    assert actual.foo.variable.data.chunks == ((3, 2, 3, 2),)
        with pytest.raises(OSError, match='no files to open'):
            open_mfdataset('foo-bar-baz-*.nc')
        with pytest.raises(ValueError, match='wild-card'):
            open_mfdataset('http://some/remote/uri')

    @requires_fsspec
    def test_open_mfdataset_no_files(self) -> None:
        pytest.importorskip('aiobotocore')
        with pytest.raises(OSError, match='no files'):
            open_mfdataset('http://some/remote/uri', engine='zarr')

    def test_open_mfdataset_2d(self) -> None:
        original = Dataset({'foo': (['x', 'y'], np.random.randn(10, 8))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                with create_tmp_file() as tmp3:
                    with create_tmp_file() as tmp4:
                        original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
                        original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
                        original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
                        original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
                        with open_mfdataset([[tmp1, tmp2], [tmp3, tmp4]], combine='nested', concat_dim=['y', 'x']) as actual:
                            assert isinstance(actual.foo.variable.data, da.Array)
                            assert actual.foo.variable.data.chunks == ((5, 5), (4, 4))
                            assert_identical(original, actual)
                        with open_mfdataset([[tmp1, tmp2], [tmp3, tmp4]], combine='nested', concat_dim=['y', 'x'], chunks={'x': 3, 'y': 2}) as actual:
                            assert actual.foo.variable.data.chunks == ((3, 2, 3, 2), (2, 2, 2, 2))

    def test_open_mfdataset_pathlib(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmps1:
            with create_tmp_file() as tmps2:
                tmp1 = Path(tmps1)
                tmp2 = Path(tmps2)
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested') as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_2d_pathlib(self) -> None:
        original = Dataset({'foo': (['x', 'y'], np.random.randn(10, 8))})
        with create_tmp_file() as tmps1:
            with create_tmp_file() as tmps2:
                with create_tmp_file() as tmps3:
                    with create_tmp_file() as tmps4:
                        tmp1 = Path(tmps1)
                        tmp2 = Path(tmps2)
                        tmp3 = Path(tmps3)
                        tmp4 = Path(tmps4)
                        original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
                        original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
                        original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
                        original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
                        with open_mfdataset([[tmp1, tmp2], [tmp3, tmp4]], combine='nested', concat_dim=['y', 'x']) as actual:
                            assert_identical(original, actual)

    def test_open_mfdataset_2(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested') as actual:
                    assert_identical(original, actual)

    def test_attrs_mfdataset(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1 = original.isel(x=slice(5))
                ds2 = original.isel(x=slice(5, 10))
                ds1.attrs['test1'] = 'foo'
                ds2.attrs['test2'] = 'bar'
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested') as actual:
                    assert actual.test1 == ds1.test1
                    with pytest.raises(AttributeError, match='no attribute'):
                        actual.test2

    def test_open_mfdataset_attrs_file(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_files(2) as (tmp1, tmp2):
            ds1 = original.isel(x=slice(5))
            ds2 = original.isel(x=slice(5, 10))
            ds1.attrs['test1'] = 'foo'
            ds2.attrs['test2'] = 'bar'
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested', attrs_file=tmp2) as actual:
                assert actual.attrs['test2'] == ds2.attrs['test2']
                assert 'test1' not in actual.attrs

    def test_open_mfdataset_attrs_file_path(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_files(2) as (tmps1, tmps2):
            tmp1 = Path(tmps1)
            tmp2 = Path(tmps2)
            ds1 = original.isel(x=slice(5))
            ds2 = original.isel(x=slice(5, 10))
            ds1.attrs['test1'] = 'foo'
            ds2.attrs['test2'] = 'bar'
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested', attrs_file=tmp2) as actual:
                assert actual.attrs['test2'] == ds2.attrs['test2']
                assert 'test1' not in actual.attrs

    def test_open_mfdataset_auto_combine(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10)), 'x': np.arange(10)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset([tmp2, tmp1], combine='by_coords') as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_raise_on_bad_combine_args(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10)), 'x': np.arange(10)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with pytest.raises(ValueError, match='`concat_dim` has no effect'):
                    open_mfdataset([tmp1, tmp2], concat_dim='x')

    def test_encoding_mfdataset(self) -> None:
        original = Dataset({'foo': ('t', np.random.randn(10)), 't': ('t', pd.date_range(start='2010-01-01', periods=10, freq='1D'))})
        original.t.encoding['units'] = 'days since 2010-01-01'
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1 = original.isel(t=slice(5))
                ds2 = original.isel(t=slice(5, 10))
                ds1.t.encoding['units'] = 'days since 2010-01-01'
                ds2.t.encoding['units'] = 'days since 2000-01-01'
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], combine='nested') as actual:
                    assert actual.t.encoding['units'] == original.t.encoding['units']
                    assert actual.t.encoding['units'] == ds1.t.encoding['units']
                    assert actual.t.encoding['units'] != ds2.t.encoding['units']

    def test_preprocess_mfdataset(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)

            def preprocess(ds):
                return ds.assign_coords(z=0)
            expected = preprocess(original)
            with open_mfdataset(tmp, preprocess=preprocess, combine='by_coords') as actual:
                assert_identical(expected, actual)

    def test_save_mfdataset_roundtrip(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested') as actual:
                    assert_identical(actual, original)

    def test_save_mfdataset_invalid(self) -> None:
        ds = Dataset()
        with pytest.raises(ValueError, match='cannot use mode'):
            save_mfdataset([ds, ds], ['same', 'same'])
        with pytest.raises(ValueError, match='same length'):
            save_mfdataset([ds, ds], ['only one path'])

    def test_save_mfdataset_invalid_dataarray(self) -> None:
        da = DataArray([1, 2])
        with pytest.raises(TypeError, match='supports writing Dataset'):
            save_mfdataset([da], ['dataarray'])

    def test_save_mfdataset_pathlib_roundtrip(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmps1:
            with create_tmp_file() as tmps2:
                tmp1 = Path(tmps1)
                tmp2 = Path(tmps2)
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset([tmp1, tmp2], concat_dim='x', combine='nested') as actual:
                    assert_identical(actual, original)

    def test_save_mfdataset_pass_kwargs(self) -> None:
        times = [0, 1]
        time = xr.DataArray(times, dims=('time',))
        test_ds = xr.Dataset()
        test_ds['time'] = time
        encoding = dict(time=dict(dtype='double'))
        unlimited_dims = ['time']
        output_path = 'test.nc'
        xr.save_mfdataset([test_ds], [output_path], encoding=encoding, unlimited_dims=unlimited_dims)

    def test_open_and_do_math(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset(tmp, combine='by_coords') as ds:
                actual = 1.0 * ds
                assert_allclose(original, actual, decode_bytes=False)

    def test_open_mfdataset_concat_dim_none(self) -> None:
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                data = Dataset({'x': 0})
                data.to_netcdf(tmp1)
                Dataset({'x': np.nan}).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], concat_dim=None, combine='nested') as actual:
                    assert_identical(data, actual)

    def test_open_mfdataset_concat_dim_default_none(self) -> None:
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                data = Dataset({'x': 0})
                data.to_netcdf(tmp1)
                Dataset({'x': np.nan}).to_netcdf(tmp2)
                with open_mfdataset([tmp1, tmp2], combine='nested') as actual:
                    assert_identical(data, actual)

    def test_open_dataset(self) -> None:
        original = Dataset({'foo': ('x', np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_dataset(tmp, chunks={'x': 5}) as actual:
                assert isinstance(actual.foo.variable.data, da.Array)
                assert actual.foo.variable.data.chunks == ((5, 5),)
                assert_identical(original, actual)
            with open_dataset(tmp, chunks=5) as actual:
                assert_identical(original, actual)
            with open_dataset(tmp) as actual:
                assert isinstance(actual.foo.variable.data, np.ndarray)
                assert_identical(original, actual)

    def test_open_single_dataset(self) -> None:
        rnddata = np.random.randn(10)
        original = Dataset({'foo': ('x', rnddata)})
        dim = DataArray([100], name='baz', dims='baz')
        expected = Dataset({'foo': (('baz', 'x'), rnddata[np.newaxis, :])}, {'baz': [100]})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset([tmp], concat_dim=dim, combine='nested') as actual:
                assert_identical(expected, actual)

    def test_open_multi_dataset(self) -> None:
        rnddata = np.random.randn(10)
        original = Dataset({'foo': ('x', rnddata)})
        dim = DataArray([100, 150], name='baz', dims='baz')
        expected = Dataset({'foo': (('baz', 'x'), np.tile(rnddata[np.newaxis, :], (2, 1)))}, {'baz': [100, 150]})
        with create_tmp_file() as tmp1, create_tmp_file() as tmp2:
            original.to_netcdf(tmp1)
            original.to_netcdf(tmp2)
            with open_mfdataset([tmp1, tmp2], concat_dim=dim, combine='nested') as actual:
                assert_identical(expected, actual)

    @pytest.mark.xfail(reason='Flaky test. Very open to contributions on fixing this')
    def test_dask_roundtrip(self) -> None:
        with create_tmp_file() as tmp:
            data = create_test_data()
            data.to_netcdf(tmp)
            chunks = {'dim1': 4, 'dim2': 4, 'dim3': 4, 'time': 10}
            with open_dataset(tmp, chunks=chunks) as dask_ds:
                assert_identical(data, dask_ds)
                with create_tmp_file() as tmp2:
                    dask_ds.to_netcdf(tmp2)
                    with open_dataset(tmp2) as on_disk:
                        assert_identical(data, on_disk)

    def test_deterministic_names(self) -> None:
        with create_tmp_file() as tmp:
            data = create_test_data()
            data.to_netcdf(tmp)
            with open_mfdataset(tmp, combine='by_coords') as ds:
                original_names = {k: v.data.name for k, v in ds.data_vars.items()}
            with open_mfdataset(tmp, combine='by_coords') as ds:
                repeat_names = {k: v.data.name for k, v in ds.data_vars.items()}
            for var_name, dask_name in original_names.items():
                assert var_name in dask_name
                assert dask_name[:13] == 'open_dataset-'
            assert original_names == repeat_names

    def test_dataarray_compute(self) -> None:
        actual = DataArray([1, 2]).chunk()
        computed = actual.compute()
        assert not actual._in_memory
        assert computed._in_memory
        assert_allclose(actual, computed, decode_bytes=False)

    def test_save_mfdataset_compute_false_roundtrip(self) -> None:
        from dask.delayed import Delayed
        original = Dataset({'foo': ('x', np.random.randn(10))}).chunk()
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as tmp1:
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as tmp2:
                delayed_obj = save_mfdataset(datasets, [tmp1, tmp2], engine=self.engine, compute=False)
                assert isinstance(delayed_obj, Delayed)
                delayed_obj.compute()
                with open_mfdataset([tmp1, tmp2], combine='nested', concat_dim='x') as actual:
                    assert_identical(actual, original)

    def test_load_dataset(self) -> None:
        with create_tmp_file() as tmp:
            original = Dataset({'foo': ('x', np.random.randn(10))})
            original.to_netcdf(tmp)
            ds = load_dataset(tmp)
            ds.to_netcdf(tmp)

    def test_load_dataarray(self) -> None:
        with create_tmp_file() as tmp:
            original = Dataset({'foo': ('x', np.random.randn(10))})
            original.to_netcdf(tmp)
            ds = load_dataarray(tmp)
            ds.to_netcdf(tmp)

    @pytest.mark.skipif(ON_WINDOWS, reason='counting number of tasks in graph fails on windows for some reason')
    def test_inline_array(self) -> None:
        with create_tmp_file() as tmp:
            original = Dataset({'foo': ('x', np.random.randn(10))})
            original.to_netcdf(tmp)
            chunks = {'time': 10}

            def num_graph_nodes(obj):
                return len(obj.__dask_graph__())
            with open_dataset(tmp, inline_array=False, chunks=chunks) as not_inlined_ds, open_dataset(tmp, inline_array=True, chunks=chunks) as inlined_ds:
                assert num_graph_nodes(inlined_ds) < num_graph_nodes(not_inlined_ds)
            with open_dataarray(tmp, inline_array=False, chunks=chunks) as not_inlined_da, open_dataarray(tmp, inline_array=True, chunks=chunks) as inlined_da:
                assert num_graph_nodes(inlined_da) < num_graph_nodes(not_inlined_da)