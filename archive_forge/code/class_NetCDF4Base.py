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
class NetCDF4Base(NetCDFBase):
    """Tests for both netCDF4-python and h5netcdf."""
    engine: T_NetcdfEngine = 'netcdf4'

    def test_open_group(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as rootgrp:
                foogrp = rootgrp.createGroup('foo')
                ds = foogrp
                ds.createDimension('time', size=10)
                x = np.arange(10)
                ds.createVariable('x', np.int32, dimensions=('time',))
                ds.variables['x'][:] = x
            expected = Dataset()
            expected['x'] = ('time', x)
            for group in ('foo', '/foo', 'foo/', '/foo/'):
                with self.open(tmp_file, group=group) as actual:
                    assert_equal(actual['x'], expected['x'])
            with pytest.raises(OSError):
                open_dataset(tmp_file, group='bar')
            with pytest.raises(ValueError, match='must be a string'):
                open_dataset(tmp_file, group=(1, 2, 3))

    def test_open_subgroup(self) -> None:
        with create_tmp_file() as tmp_file:
            rootgrp = nc4.Dataset(tmp_file, 'w')
            foogrp = rootgrp.createGroup('foo')
            bargrp = foogrp.createGroup('bar')
            ds = bargrp
            ds.createDimension('time', size=10)
            x = np.arange(10)
            ds.createVariable('x', np.int32, dimensions=('time',))
            ds.variables['x'][:] = x
            rootgrp.close()
            expected = Dataset()
            expected['x'] = ('time', x)
            for group in ('foo/bar', '/foo/bar', 'foo/bar/', '/foo/bar/'):
                with self.open(tmp_file, group=group) as actual:
                    assert_equal(actual['x'], expected['x'])

    def test_write_groups(self) -> None:
        data1 = create_test_data()
        data2 = data1 * 2
        with create_tmp_file() as tmp_file:
            self.save(data1, tmp_file, group='data/1')
            self.save(data2, tmp_file, group='data/2', mode='a')
            with self.open(tmp_file, group='data/1') as actual1:
                assert_identical(data1, actual1)
            with self.open(tmp_file, group='data/2') as actual2:
                assert_identical(data2, actual2)

    @pytest.mark.parametrize('input_strings, is_bytes', [([b'foo', b'bar', b'baz'], True), (['foo', 'bar', 'baz'], False), (['foó', 'bár', 'baź'], False)])
    def test_encoding_kwarg_vlen_string(self, input_strings: list[str], is_bytes: bool) -> None:
        original = Dataset({'x': input_strings})
        expected_string = ['foo', 'bar', 'baz'] if is_bytes else input_strings
        expected = Dataset({'x': expected_string})
        kwargs = dict(encoding={'x': {'dtype': str}})
        with self.roundtrip(original, save_kwargs=kwargs) as actual:
            assert actual['x'].encoding['dtype'] == '<U3'
            assert actual['x'].dtype == '<U3'
            assert_identical(actual, expected)

    @pytest.mark.parametrize('fill_value', ['XXX', '', 'bár'])
    def test_roundtrip_string_with_fill_value_vlen(self, fill_value: str) -> None:
        values = np.array(['ab', 'cdef', np.nan], dtype=object)
        expected = Dataset({'x': ('t', values)})
        original = Dataset({'x': ('t', values, {}, {'_FillValue': fill_value})})
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)
        original = Dataset({'x': ('t', values, {}, {'_FillValue': ''})})
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_character_array(self) -> None:
        with create_tmp_file() as tmp_file:
            values = np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype='S')
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('x', 2)
                nc.createDimension('string3', 3)
                v = nc.createVariable('x', np.dtype('S1'), ('x', 'string3'))
                v[:] = values
            values = np.array(['abc', 'def'], dtype='S')
            expected = Dataset({'x': ('x', values)})
            with open_dataset(tmp_file) as actual:
                assert_identical(expected, actual)
                with self.roundtrip(actual) as roundtripped:
                    assert_identical(expected, roundtripped)

    def test_default_to_char_arrays(self) -> None:
        data = Dataset({'x': np.array(['foo', 'zzzz'], dtype='S')})
        with self.roundtrip(data) as actual:
            assert_identical(data, actual)
            assert actual['x'].dtype == np.dtype('S4')

    def test_open_encodings(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as ds:
                ds.createDimension('time', size=10)
                ds.createVariable('time', np.int32, dimensions=('time',))
                units = 'days since 1999-01-01'
                ds.variables['time'].setncattr('units', units)
                ds.variables['time'][:] = np.arange(10) + 4
            expected = Dataset()
            time = pd.date_range('1999-01-05', periods=10)
            encoding = {'units': units, 'dtype': np.dtype('int32')}
            expected['time'] = ('time', time, {}, encoding)
            with open_dataset(tmp_file) as actual:
                assert_equal(actual['time'], expected['time'])
                actual_encoding = {k: v for k, v in actual['time'].encoding.items() if k in expected['time'].encoding}
                assert actual_encoding == expected['time'].encoding

    def test_dump_encodings(self) -> None:
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = dict(encoding={'x': {'zlib': True}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding['zlib']

    def test_dump_and_open_encodings(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as ds:
                ds.createDimension('time', size=10)
                ds.createVariable('time', np.int32, dimensions=('time',))
                units = 'days since 1999-01-01'
                ds.variables['time'].setncattr('units', units)
                ds.variables['time'][:] = np.arange(10) + 4
            with open_dataset(tmp_file) as xarray_dataset:
                with create_tmp_file() as tmp_file2:
                    xarray_dataset.to_netcdf(tmp_file2)
                    with nc4.Dataset(tmp_file2, 'r') as ds:
                        assert ds.variables['time'].getncattr('units') == units
                        assert_array_equal(ds.variables['time'], np.arange(10) + 4)

    def test_compression_encoding_legacy(self) -> None:
        data = create_test_data()
        data['var2'].encoding.update({'zlib': True, 'chunksizes': (5, 5), 'fletcher32': True, 'shuffle': True, 'original_shape': data.var2.shape})
        with self.roundtrip(data) as actual:
            for k, v in data['var2'].encoding.items():
                assert v == actual['var2'].encoding[k]
        expected = data.isel(dim1=0)
        with self.roundtrip(expected) as actual:
            assert_equal(expected, actual)

    def test_encoding_kwarg_compression(self) -> None:
        ds = Dataset({'x': np.arange(10.0)})
        encoding = dict(dtype='f4', zlib=True, complevel=9, fletcher32=True, chunksizes=(5,), shuffle=True)
        kwargs = dict(encoding=dict(x=encoding))
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert_equal(actual, ds)
            assert actual.x.encoding['dtype'] == 'f4'
            assert actual.x.encoding['zlib']
            assert actual.x.encoding['complevel'] == 9
            assert actual.x.encoding['fletcher32']
            assert actual.x.encoding['chunksizes'] == (5,)
            assert actual.x.encoding['shuffle']
        assert ds.x.encoding == {}

    def test_keep_chunksizes_if_no_original_shape(self) -> None:
        ds = Dataset({'x': [1, 2, 3]})
        chunksizes = (2,)
        ds.variables['x'].encoding = {'chunksizes': chunksizes}
        with self.roundtrip(ds) as actual:
            assert_identical(ds, actual)
            assert_array_equal(ds['x'].encoding['chunksizes'], actual['x'].encoding['chunksizes'])

    def test_preferred_chunks_is_present(self) -> None:
        ds = Dataset({'x': [1, 2, 3]})
        chunksizes = (2,)
        ds.variables['x'].encoding = {'chunksizes': chunksizes}
        with self.roundtrip(ds) as actual:
            assert actual['x'].encoding['preferred_chunks'] == {'x': 2}

    @requires_dask
    def test_auto_chunking_is_based_on_disk_chunk_sizes(self) -> None:
        x_size = y_size = 1000
        y_chunksize = y_size
        x_chunksize = 10
        with dask.config.set({'array.chunk-size': '100KiB'}):
            with self.chunked_roundtrip((1, y_size, x_size), (1, y_chunksize, x_chunksize), open_kwargs={'chunks': 'auto'}) as ds:
                t_chunks, y_chunks, x_chunks = ds['image'].data.chunks
                assert all(np.asanyarray(y_chunks) == y_chunksize)
                assert all(np.asanyarray(x_chunks) % x_chunksize == 0)

    @requires_dask
    def test_base_chunking_uses_disk_chunk_sizes(self) -> None:
        x_size = y_size = 1000
        y_chunksize = y_size
        x_chunksize = 10
        with self.chunked_roundtrip((1, y_size, x_size), (1, y_chunksize, x_chunksize), open_kwargs={'chunks': {}}) as ds:
            for chunksizes, expected in zip(ds['image'].data.chunks, (1, y_chunksize, x_chunksize)):
                assert all(np.asanyarray(chunksizes) == expected)

    @contextlib.contextmanager
    def chunked_roundtrip(self, array_shape: tuple[int, int, int], chunk_sizes: tuple[int, int, int], open_kwargs: dict[str, Any] | None=None) -> Generator[Dataset, None, None]:
        t_size, y_size, x_size = array_shape
        t_chunksize, y_chunksize, x_chunksize = chunk_sizes
        image = xr.DataArray(np.arange(t_size * x_size * y_size, dtype=np.int16).reshape((t_size, y_size, x_size)), dims=['t', 'y', 'x'])
        image.encoding = {'chunksizes': (t_chunksize, y_chunksize, x_chunksize)}
        dataset = xr.Dataset(dict(image=image))
        with self.roundtrip(dataset, open_kwargs=open_kwargs) as ds:
            yield ds

    def test_preferred_chunks_are_disk_chunk_sizes(self) -> None:
        x_size = y_size = 1000
        y_chunksize = y_size
        x_chunksize = 10
        with self.chunked_roundtrip((1, y_size, x_size), (1, y_chunksize, x_chunksize)) as ds:
            assert ds['image'].encoding['preferred_chunks'] == {'t': 1, 'y': y_chunksize, 'x': x_chunksize}

    def test_encoding_chunksizes_unlimited(self) -> None:
        ds = Dataset({'x': [1, 2, 3], 'y': ('x', [2, 3, 4])})
        ds.variables['x'].encoding = {'zlib': False, 'shuffle': False, 'complevel': 0, 'fletcher32': False, 'contiguous': False, 'chunksizes': (2 ** 20,), 'original_shape': (3,)}
        with self.roundtrip(ds) as actual:
            assert_equal(ds, actual)

    def test_mask_and_scale(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('t', 5)
                nc.createVariable('x', 'int16', ('t',), fill_value=-1)
                v = nc.variables['x']
                v.set_auto_maskandscale(False)
                v.add_offset = 10
                v.scale_factor = 0.1
                v[:] = np.array([-1, -1, 0, 1, 2])
                dtype = type(v.scale_factor)
            with nc4.Dataset(tmp_file, mode='r') as nc:
                expected = np.ma.array([-1, -1, 10, 10.1, 10.2], mask=[True, True, False, False, False])
                actual = nc.variables['x'][:]
                assert_array_equal(expected, actual)
            with open_dataset(tmp_file) as ds:
                expected = create_masked_and_scaled_data(np.dtype(dtype))
                assert_identical(expected, ds)

    def test_0dimensional_variable(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                v = nc.createVariable('x', 'int16')
                v[...] = 123
            with open_dataset(tmp_file) as ds:
                expected = Dataset({'x': ((), 123)})
                assert_identical(expected, ds)

    def test_read_variable_len_strings(self) -> None:
        with create_tmp_file() as tmp_file:
            values = np.array(['foo', 'bar', 'baz'], dtype=object)
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('x', 3)
                v = nc.createVariable('x', str, ('x',))
                v[:] = values
            expected = Dataset({'x': ('x', values)})
            for kwargs in [{}, {'decode_cf': True}]:
                with open_dataset(tmp_file, **cast(dict, kwargs)) as actual:
                    assert_identical(expected, actual)

    def test_encoding_unlimited_dims(self) -> None:
        ds = Dataset({'x': ('y', np.arange(10.0))})
        with self.roundtrip(ds, save_kwargs=dict(unlimited_dims=['y'])) as actual:
            assert actual.encoding['unlimited_dims'] == set('y')
            assert_equal(ds, actual)
        ds.encoding = {'unlimited_dims': ['y']}
        with self.roundtrip(ds) as actual:
            assert actual.encoding['unlimited_dims'] == set('y')
            assert_equal(ds, actual)

    def test_raise_on_forward_slashes_in_names(self) -> None:
        data_vars: list[dict[str, Any]] = [{'PASS/FAIL': (['PASSFAIL'], np.array([0]))}, {'PASS/FAIL': np.array([0])}, {'PASSFAIL': (['PASS/FAIL'], np.array([0]))}]
        for dv in data_vars:
            ds = Dataset(data_vars=dv)
            with pytest.raises(ValueError, match="Forward slashes '/' are not allowed"):
                with self.roundtrip(ds):
                    pass

    @requires_netCDF4
    def test_encoding_enum__no_fill_value(self):
        with create_tmp_file() as tmp_file:
            cloud_type_dict = {'clear': 0, 'cloudy': 1}
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('time', size=2)
                cloud_type = nc.createEnumType('u1', 'cloud_type', cloud_type_dict)
                v = nc.createVariable('clouds', cloud_type, 'time', fill_value=None)
                v[:] = 1
            with open_dataset(tmp_file) as original:
                save_kwargs = {}
                if self.engine == 'h5netcdf':
                    save_kwargs['invalid_netcdf'] = True
                with self.roundtrip(original, save_kwargs=save_kwargs) as actual:
                    assert_equal(original, actual)
                    assert actual.clouds.encoding['dtype'].metadata['enum'] == cloud_type_dict
                    if self.engine != 'h5netcdf':
                        assert actual.clouds.encoding['dtype'].metadata['enum_name'] == 'cloud_type'

    @requires_netCDF4
    def test_encoding_enum__multiple_variable_with_enum(self):
        with create_tmp_file() as tmp_file:
            cloud_type_dict = {'clear': 0, 'cloudy': 1, 'missing': 255}
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('time', size=2)
                cloud_type = nc.createEnumType('u1', 'cloud_type', cloud_type_dict)
                nc.createVariable('clouds', cloud_type, 'time', fill_value=255)
                nc.createVariable('tifa', cloud_type, 'time', fill_value=255)
            with open_dataset(tmp_file) as original:
                save_kwargs = {}
                if self.engine == 'h5netcdf':
                    save_kwargs['invalid_netcdf'] = True
                with self.roundtrip(original, save_kwargs=save_kwargs) as actual:
                    assert_equal(original, actual)
                    assert actual.clouds.encoding['dtype'] == actual.tifa.encoding['dtype']
                    assert actual.clouds.encoding['dtype'].metadata == actual.tifa.encoding['dtype'].metadata
                    assert actual.clouds.encoding['dtype'].metadata['enum'] == cloud_type_dict
                    if self.engine != 'h5netcdf':
                        assert actual.clouds.encoding['dtype'].metadata['enum_name'] == 'cloud_type'

    @requires_netCDF4
    def test_encoding_enum__error_multiple_variable_with_changing_enum(self):
        """
        Given 2 variables, if they share the same enum type,
        the 2 enum definition should be identical.
        """
        with create_tmp_file() as tmp_file:
            cloud_type_dict = {'clear': 0, 'cloudy': 1, 'missing': 255}
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('time', size=2)
                cloud_type = nc.createEnumType('u1', 'cloud_type', cloud_type_dict)
                nc.createVariable('clouds', cloud_type, 'time', fill_value=255)
                nc.createVariable('tifa', cloud_type, 'time', fill_value=255)
            with open_dataset(tmp_file) as original:
                assert original.clouds.encoding['dtype'].metadata == original.tifa.encoding['dtype'].metadata
                modified_enum = original.clouds.encoding['dtype'].metadata['enum']
                modified_enum.update({'neblig': 2})
                original.clouds.encoding['dtype'] = np.dtype('u1', metadata={'enum': modified_enum, 'enum_name': 'cloud_type'})
                if self.engine != 'h5netcdf':
                    with pytest.raises(ValueError, match='Cannot save variable .* because an enum `cloud_type` already exists in the Dataset .*'):
                        with self.roundtrip(original):
                            pass