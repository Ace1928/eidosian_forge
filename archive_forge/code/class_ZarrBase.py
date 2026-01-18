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
@requires_zarr
class ZarrBase(CFEncodedBase):
    DIMENSION_KEY = '_ARRAY_DIMENSIONS'
    zarr_version = 2
    version_kwargs: dict[str, Any] = {}

    def create_zarr_target(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def create_store(self):
        with self.create_zarr_target() as store_target:
            yield backends.ZarrStore.open_group(store_target, mode='w', **self.version_kwargs)

    def save(self, dataset, store_target, **kwargs):
        return dataset.to_zarr(store=store_target, **kwargs, **self.version_kwargs)

    @contextlib.contextmanager
    def open(self, store_target, **kwargs):
        with xr.open_dataset(store_target, engine='zarr', **kwargs, **self.version_kwargs) as ds:
            yield ds

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with self.create_zarr_target() as store_target:
            self.save(data, store_target, **save_kwargs)
            with self.open(store_target, **open_kwargs) as ds:
                yield ds

    @pytest.mark.parametrize('consolidated', [False, True, None])
    def test_roundtrip_consolidated(self, consolidated) -> None:
        if consolidated and self.zarr_version > 2:
            pytest.xfail('consolidated metadata is not supported for zarr v3 yet')
        expected = create_test_data()
        with self.roundtrip(expected, save_kwargs={'consolidated': consolidated}, open_kwargs={'backend_kwargs': {'consolidated': consolidated}}) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            assert_identical(expected, actual)

    def test_read_non_consolidated_warning(self) -> None:
        if self.zarr_version > 2:
            pytest.xfail('consolidated metadata is not supported for zarr v3 yet')
        expected = create_test_data()
        with self.create_zarr_target() as store:
            expected.to_zarr(store, consolidated=False, **self.version_kwargs)
            with pytest.warns(RuntimeWarning, match='Failed to open Zarr store with consolidated'):
                with xr.open_zarr(store, **self.version_kwargs) as ds:
                    assert_identical(ds, expected)

    def test_non_existent_store(self) -> None:
        with pytest.raises(FileNotFoundError, match='No such file or directory:'):
            xr.open_zarr(f'{uuid.uuid4()}')

    def test_with_chunkstore(self) -> None:
        expected = create_test_data()
        with self.create_zarr_target() as store_target, self.create_zarr_target() as chunk_store:
            save_kwargs = {'chunk_store': chunk_store}
            self.save(expected, store_target, **save_kwargs)
            assert len(chunk_store) > 0
            open_kwargs = {'backend_kwargs': {'chunk_store': chunk_store}}
            with self.open(store_target, **open_kwargs) as ds:
                assert_equal(ds, expected)

    @requires_dask
    def test_auto_chunk(self) -> None:
        original = create_test_data().chunk()
        with self.roundtrip(original, open_kwargs={'chunks': None}) as actual:
            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
                assert v.chunks is None
        with self.roundtrip(original, open_kwargs={'chunks': {}}) as actual:
            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
                assert v.chunks == original[k].chunks

    @requires_dask
    @pytest.mark.filterwarnings('ignore:The specified chunks separate:UserWarning')
    def test_manual_chunk(self) -> None:
        original = create_test_data().chunk({'dim1': 3, 'dim2': 4, 'dim3': 3})
        open_kwargs: dict[str, Any] = {'chunks': None}
        with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
                assert v.chunks is None
        for i in range(2, 6):
            rechunked = original.chunk(chunks=i)
            open_kwargs = {'chunks': i}
            with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
                for k, v in actual.variables.items():
                    assert v._in_memory == (k in actual.dims)
                    assert v.chunks == rechunked[k].chunks
        chunks = {'dim1': 2, 'dim2': 3, 'dim3': 5}
        rechunked = original.chunk(chunks=chunks)
        open_kwargs = {'chunks': chunks, 'backend_kwargs': {'overwrite_encoded_chunks': True}}
        with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
            for k, v in actual.variables.items():
                assert v.chunks == rechunked[k].chunks
            with self.roundtrip(actual) as auto:
                for k, v in actual.variables.items():
                    assert v.chunks == rechunked[k].chunks
                assert_identical(actual, auto)
                assert_identical(actual.load(), auto.load())

    @requires_dask
    def test_warning_on_bad_chunks(self) -> None:
        original = create_test_data().chunk({'dim1': 4, 'dim2': 3, 'dim3': 3})
        bad_chunks = (2, {'dim2': (3, 3, 2, 1)})
        for chunks in bad_chunks:
            kwargs = {'chunks': chunks}
            with pytest.warns(UserWarning):
                with self.roundtrip(original, open_kwargs=kwargs) as actual:
                    for k, v in actual.variables.items():
                        assert v._in_memory == (k in actual.dims)
        good_chunks: tuple[dict[str, Any], ...] = ({'dim2': 3}, {'dim3': (6, 4)}, {})
        for chunks in good_chunks:
            kwargs = {'chunks': chunks}
            with assert_no_warnings():
                with self.roundtrip(original, open_kwargs=kwargs) as actual:
                    for k, v in actual.variables.items():
                        assert v._in_memory == (k in actual.dims)

    @requires_dask
    def test_deprecate_auto_chunk(self) -> None:
        original = create_test_data().chunk()
        with pytest.raises(TypeError):
            with self.roundtrip(original, open_kwargs={'auto_chunk': True}) as actual:
                for k, v in actual.variables.items():
                    assert v._in_memory == (k in actual.dims)
                    assert v.chunks == original[k].chunks
        with pytest.raises(TypeError):
            with self.roundtrip(original, open_kwargs={'auto_chunk': False}) as actual:
                for k, v in actual.variables.items():
                    assert v._in_memory == (k in actual.dims)
                    assert v.chunks is None

    @requires_dask
    def test_write_uneven_dask_chunks(self) -> None:
        original = create_test_data().chunk({'dim1': 3, 'dim2': 4, 'dim3': 3})
        with self.roundtrip(original, open_kwargs={'chunks': {}}) as actual:
            for k, v in actual.data_vars.items():
                assert v.chunks == actual[k].chunks

    def test_chunk_encoding(self) -> None:
        data = create_test_data()
        chunks = (5, 5)
        data['var2'].encoding.update({'chunks': chunks})
        with self.roundtrip(data) as actual:
            assert chunks == actual['var2'].encoding['chunks']
        data['var2'].encoding.update({'chunks': (5, 4.5)})
        with pytest.raises(TypeError):
            with self.roundtrip(data) as actual:
                pass

    @requires_dask
    @pytest.mark.skipif(ON_WINDOWS, reason='Very flaky on Windows CI. Can re-enable assuming it starts consistently passing.')
    def test_chunk_encoding_with_dask(self) -> None:
        ds = xr.DataArray(np.arange(12), dims='x', name='var1').to_dataset()
        ds_chunk4 = ds.chunk({'x': 4})
        with self.roundtrip(ds_chunk4) as actual:
            assert (4,) == actual['var1'].encoding['chunks']
        ds_chunk_irreg = ds.chunk({'x': (5, 4, 3)})
        with pytest.raises(ValueError, match='uniform chunk sizes.'):
            with self.roundtrip(ds_chunk_irreg) as actual:
                pass
        badenc = ds.chunk({'x': 4})
        badenc.var1.encoding['chunks'] = (6,)
        with pytest.raises(ValueError, match="named 'var1' would overlap"):
            with self.roundtrip(badenc) as actual:
                pass
        with self.roundtrip(badenc, save_kwargs={'safe_chunks': False}) as actual:
            pass
        goodenc = ds.chunk({'x': 4})
        goodenc.var1.encoding['chunks'] = (2,)
        with self.roundtrip(goodenc) as actual:
            pass
        goodenc = ds.chunk({'x': (3, 3, 6)})
        goodenc.var1.encoding['chunks'] = (3,)
        with self.roundtrip(goodenc) as actual:
            pass
        goodenc = ds.chunk({'x': (3, 6, 3)})
        goodenc.var1.encoding['chunks'] = (3,)
        with self.roundtrip(goodenc) as actual:
            pass
        ds_chunk_irreg = ds.chunk({'x': (5, 5, 2)})
        with self.roundtrip(ds_chunk_irreg) as actual:
            assert (5,) == actual['var1'].encoding['chunks']
        with self.roundtrip(ds_chunk_irreg) as original:
            with self.roundtrip(original) as actual:
                assert_identical(original, actual)
        badenc = ds.chunk({'x': (3, 5, 3, 1)})
        badenc.var1.encoding['chunks'] = (3,)
        with pytest.raises(ValueError, match='would overlap multiple dask chunks'):
            with self.roundtrip(badenc) as actual:
                pass
        for chunk_enc in (4, (4,)):
            ds_chunk4['var1'].encoding.update({'chunks': chunk_enc})
            with self.roundtrip(ds_chunk4) as actual:
                assert (4,) == actual['var1'].encoding['chunks']
        ds_chunk4['var1'].encoding.update({'chunks': 5})
        with pytest.raises(ValueError, match="named 'var1' would overlap"):
            with self.roundtrip(ds_chunk4) as actual:
                pass
        with self.roundtrip(ds_chunk4, save_kwargs={'safe_chunks': False}) as actual:
            pass

    def test_drop_encoding(self):
        with open_example_dataset('example_1.nc') as ds:
            encodings = {v: {**ds[v].encoding} for v in ds.data_vars}
            with self.create_zarr_target() as store:
                ds.to_zarr(store, encoding=encodings)

    def test_hidden_zarr_keys(self) -> None:
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            zarr_group = store.ds
            for var in expected.variables.keys():
                dims = zarr_group[var].attrs[self.DIMENSION_KEY]
                assert dims == list(expected[var].dims)
            with xr.decode_cf(store):
                for var in expected.variables.keys():
                    assert self.DIMENSION_KEY not in expected[var].attrs
            del zarr_group.var2.attrs[self.DIMENSION_KEY]
            with pytest.raises(KeyError):
                with xr.decode_cf(store):
                    pass

    @pytest.mark.parametrize('group', [None, 'group1'])
    def test_write_persistence_modes(self, group) -> None:
        original = create_test_data()
        with self.roundtrip(original, save_kwargs={'mode': 'w', 'group': group}, open_kwargs={'group': group}) as actual:
            assert_identical(original, actual)
        with self.roundtrip(original, save_kwargs={'mode': 'w-', 'group': group}, open_kwargs={'group': group}) as actual:
            assert_identical(original, actual)
        with self.create_zarr_target() as store:
            self.save(original, store)
            self.save(original, store, mode='w', group=group)
            with self.open(store, group=group) as actual:
                assert_identical(original, actual)
                with pytest.raises(ValueError):
                    self.save(original, store, mode='w-')
        with self.roundtrip(original, save_kwargs={'mode': 'a', 'group': group}, open_kwargs={'group': group}) as actual:
            assert_identical(original, actual)
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', group=group, **self.version_kwargs)
            ds_to_append.to_zarr(store_target, append_dim='time', group=group, **self.version_kwargs)
            original = xr.concat([ds, ds_to_append], dim='time')
            actual = xr.open_dataset(store_target, group=group, engine='zarr', **self.version_kwargs)
            assert_identical(original, actual)

    def test_compressor_encoding(self) -> None:
        original = create_test_data()
        import zarr
        blosc_comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        save_kwargs = dict(encoding={'var1': {'compressor': blosc_comp}})
        with self.roundtrip(original, save_kwargs=save_kwargs) as ds:
            actual = ds['var1'].encoding['compressor']
            assert actual.get_config() == blosc_comp.get_config()

    def test_group(self) -> None:
        original = create_test_data()
        group = 'some/random/path'
        with self.roundtrip(original, save_kwargs={'group': group}, open_kwargs={'group': group}) as actual:
            assert_identical(original, actual)

    def test_zarr_mode_w_overwrites_encoding(self) -> None:
        import zarr
        data = Dataset({'foo': ('x', [1.0, 1.0, 1.0])})
        with self.create_zarr_target() as store:
            data.to_zarr(store, **self.version_kwargs, encoding={'foo': {'add_offset': 1}})
            np.testing.assert_equal(zarr.open_group(store, **self.version_kwargs)['foo'], data.foo.data - 1)
            data.to_zarr(store, **self.version_kwargs, encoding={'foo': {'add_offset': 0}}, mode='w')
            np.testing.assert_equal(zarr.open_group(store, **self.version_kwargs)['foo'], data.foo.data)

    def test_encoding_kwarg_fixed_width_string(self) -> None:
        pass

    def test_dataset_caching(self) -> None:
        super().test_dataset_caching()

    def test_append_write(self) -> None:
        super().test_append_write()

    def test_append_with_mode_rplus_success(self) -> None:
        original = Dataset({'foo': ('x', [1])})
        modified = Dataset({'foo': ('x', [2])})
        with self.create_zarr_target() as store:
            original.to_zarr(store, **self.version_kwargs)
            modified.to_zarr(store, mode='r+', **self.version_kwargs)
            with self.open(store) as actual:
                assert_identical(actual, modified)

    def test_append_with_mode_rplus_fails(self) -> None:
        original = Dataset({'foo': ('x', [1])})
        modified = Dataset({'bar': ('x', [2])})
        with self.create_zarr_target() as store:
            original.to_zarr(store, **self.version_kwargs)
            with pytest.raises(ValueError, match='dataset contains non-pre-existing variables'):
                modified.to_zarr(store, mode='r+', **self.version_kwargs)

    def test_append_with_invalid_dim_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', **self.version_kwargs)
            with pytest.raises(ValueError, match='does not match any existing dataset dimensions'):
                ds_to_append.to_zarr(store_target, append_dim='notvalid', **self.version_kwargs)

    def test_append_with_no_dims_raises(self) -> None:
        with self.create_zarr_target() as store_target:
            Dataset({'foo': ('x', [1])}).to_zarr(store_target, mode='w', **self.version_kwargs)
            with pytest.raises(ValueError, match='different dimension names'):
                Dataset({'foo': ('y', [2])}).to_zarr(store_target, mode='a', **self.version_kwargs)

    def test_append_with_append_dim_not_set_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', **self.version_kwargs)
            with pytest.raises(ValueError, match='different dimension sizes'):
                ds_to_append.to_zarr(store_target, mode='a', **self.version_kwargs)

    def test_append_with_mode_not_a_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', **self.version_kwargs)
            with pytest.raises(ValueError, match='cannot set append_dim unless'):
                ds_to_append.to_zarr(store_target, mode='w', append_dim='time', **self.version_kwargs)

    def test_append_with_existing_encoding_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', **self.version_kwargs)
            with pytest.raises(ValueError, match='but encoding was provided'):
                ds_to_append.to_zarr(store_target, append_dim='time', encoding={'da': {'compressor': None}}, **self.version_kwargs)

    @pytest.mark.parametrize('dtype', ['U', 'S'])
    def test_append_string_length_mismatch_raises(self, dtype) -> None:
        ds, ds_to_append = create_append_string_length_mismatch_test_data(dtype)
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', **self.version_kwargs)
            with pytest.raises(ValueError, match='Mismatched dtypes for variable'):
                ds_to_append.to_zarr(store_target, append_dim='time', **self.version_kwargs)

    def test_check_encoding_is_consistent_after_append(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            import zarr
            compressor = zarr.Blosc()
            encoding = {'da': {'compressor': compressor}}
            ds.to_zarr(store_target, mode='w', encoding=encoding, **self.version_kwargs)
            ds_to_append.to_zarr(store_target, append_dim='time', **self.version_kwargs)
            actual_ds = xr.open_dataset(store_target, engine='zarr', **self.version_kwargs)
            actual_encoding = actual_ds['da'].encoding['compressor']
            assert actual_encoding.get_config() == compressor.get_config()
            assert_identical(xr.open_dataset(store_target, engine='zarr', **self.version_kwargs).compute(), xr.concat([ds, ds_to_append], dim='time'))

    def test_append_with_new_variable(self) -> None:
        ds, ds_to_append, ds_with_new_var = create_append_test_data()
        with self.create_zarr_target() as store_target:
            xr.concat([ds, ds_to_append], dim='time').to_zarr(store_target, mode='w', **self.version_kwargs)
            ds_with_new_var.to_zarr(store_target, mode='a', **self.version_kwargs)
            combined = xr.concat([ds, ds_to_append], dim='time')
            combined['new_var'] = ds_with_new_var['new_var']
            assert_identical(combined, xr.open_dataset(store_target, engine='zarr', **self.version_kwargs))

    def test_append_with_append_dim_no_overwrite(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode='w', **self.version_kwargs)
            original = xr.concat([ds, ds_to_append], dim='time')
            original2 = xr.concat([original, ds_to_append], dim='time')
            lon = ds_to_append.lon.to_numpy().copy()
            lon[:] = -999
            ds_to_append['lon'] = lon
            ds_to_append.to_zarr(store_target, mode='a-', append_dim='time', **self.version_kwargs)
            actual = xr.open_dataset(store_target, engine='zarr', **self.version_kwargs)
            assert_identical(original, actual)
            ds_to_append.to_zarr(store_target, append_dim='time', **self.version_kwargs)
            actual = xr.open_dataset(store_target, engine='zarr', **self.version_kwargs)
            lon = original2.lon.to_numpy().copy()
            lon[:] = -999
            original2['lon'] = lon
            assert_identical(original2, actual)

    @requires_dask
    def test_to_zarr_compute_false_roundtrip(self) -> None:
        from dask.delayed import Delayed
        original = create_test_data().chunk()
        with self.create_zarr_target() as store:
            delayed_obj = self.save(original, store, compute=False)
            assert isinstance(delayed_obj, Delayed)
            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(original, actual)
            delayed_obj.compute()
            with self.open(store) as actual:
                assert_identical(original, actual)

    @requires_dask
    def test_to_zarr_append_compute_false_roundtrip(self) -> None:
        from dask.delayed import Delayed
        ds, ds_to_append, _ = create_append_test_data()
        ds, ds_to_append = (ds.chunk(), ds_to_append.chunk())
        with pytest.warns(SerializationWarning):
            with self.create_zarr_target() as store:
                delayed_obj = self.save(ds, store, compute=False, mode='w')
                assert isinstance(delayed_obj, Delayed)
                with pytest.raises(AssertionError):
                    with self.open(store) as actual:
                        assert_identical(ds, actual)
                delayed_obj.compute()
                with self.open(store) as actual:
                    assert_identical(ds, actual)
                delayed_obj = self.save(ds_to_append, store, compute=False, append_dim='time')
                assert isinstance(delayed_obj, Delayed)
                with pytest.raises(AssertionError):
                    with self.open(store) as actual:
                        assert_identical(xr.concat([ds, ds_to_append], dim='time'), actual)
                delayed_obj.compute()
                with self.open(store) as actual:
                    assert_identical(xr.concat([ds, ds_to_append], dim='time'), actual)

    @pytest.mark.parametrize('chunk', [False, True])
    def test_save_emptydim(self, chunk) -> None:
        if chunk and (not has_dask):
            pytest.skip('requires dask')
        ds = Dataset({'x': (('a', 'b'), np.empty((5, 0))), 'y': ('a', [1, 2, 5, 8, 9])})
        if chunk:
            ds = ds.chunk({})
        with self.roundtrip(ds) as ds_reload:
            assert_identical(ds, ds_reload)

    @requires_dask
    def test_no_warning_from_open_emptydim_with_chunks(self) -> None:
        ds = Dataset({'x': (('a', 'b'), np.empty((5, 0)))}).chunk({'a': 1})
        with assert_no_warnings():
            with self.roundtrip(ds, open_kwargs=dict(chunks={'a': 1})) as ds_reload:
                assert_identical(ds, ds_reload)

    @pytest.mark.parametrize('consolidated', [False, True, None])
    @pytest.mark.parametrize('compute', [False, True])
    @pytest.mark.parametrize('use_dask', [False, True])
    @pytest.mark.parametrize('write_empty', [False, True, None])
    def test_write_region(self, consolidated, compute, use_dask, write_empty) -> None:
        if (use_dask or not compute) and (not has_dask):
            pytest.skip('requires dask')
        if consolidated and self.zarr_version > 2:
            pytest.xfail('consolidated metadata is not supported for zarr v3 yet')
        zeros = Dataset({'u': (('x',), np.zeros(10))})
        nonzeros = Dataset({'u': (('x',), np.arange(1, 11))})
        if use_dask:
            zeros = zeros.chunk(2)
            nonzeros = nonzeros.chunk(2)
        with self.create_zarr_target() as store:
            zeros.to_zarr(store, consolidated=consolidated, compute=compute, encoding={'u': dict(chunks=2)}, **self.version_kwargs)
            if compute:
                with xr.open_zarr(store, consolidated=consolidated, **self.version_kwargs) as actual:
                    assert_identical(actual, zeros)
            for i in range(0, 10, 2):
                region = {'x': slice(i, i + 2)}
                nonzeros.isel(region).to_zarr(store, region=region, consolidated=consolidated, write_empty_chunks=write_empty, **self.version_kwargs)
            with xr.open_zarr(store, consolidated=consolidated, **self.version_kwargs) as actual:
                assert_identical(actual, nonzeros)

    @pytest.mark.parametrize('mode', [None, 'r+', 'a'])
    def test_write_region_mode(self, mode) -> None:
        zeros = Dataset({'u': (('x',), np.zeros(10))})
        nonzeros = Dataset({'u': (('x',), np.arange(1, 11))})
        with self.create_zarr_target() as store:
            zeros.to_zarr(store, **self.version_kwargs)
            for region in [{'x': slice(5)}, {'x': slice(5, 10)}]:
                nonzeros.isel(region).to_zarr(store, region=region, mode=mode, **self.version_kwargs)
            with xr.open_zarr(store, **self.version_kwargs) as actual:
                assert_identical(actual, nonzeros)

    @requires_dask
    def test_write_preexisting_override_metadata(self) -> None:
        """Metadata should be overridden if mode="a" but not in mode="r+"."""
        original = Dataset({'u': (('x',), np.zeros(10), {'variable': 'original'})}, attrs={'global': 'original'})
        both_modified = Dataset({'u': (('x',), np.ones(10), {'variable': 'modified'})}, attrs={'global': 'modified'})
        global_modified = Dataset({'u': (('x',), np.ones(10), {'variable': 'original'})}, attrs={'global': 'modified'})
        only_new_data = Dataset({'u': (('x',), np.ones(10), {'variable': 'original'})}, attrs={'global': 'original'})
        with self.create_zarr_target() as store:
            original.to_zarr(store, compute=False, **self.version_kwargs)
            both_modified.to_zarr(store, mode='a', **self.version_kwargs)
            with self.open(store) as actual:
                assert_identical(actual, global_modified)
        with self.create_zarr_target() as store:
            original.to_zarr(store, compute=False, **self.version_kwargs)
            both_modified.to_zarr(store, mode='r+', **self.version_kwargs)
            with self.open(store) as actual:
                assert_identical(actual, only_new_data)
        with self.create_zarr_target() as store:
            original.to_zarr(store, compute=False, **self.version_kwargs)
            both_modified.to_zarr(store, region={'x': slice(None)}, **self.version_kwargs)
            with self.open(store) as actual:
                assert_identical(actual, only_new_data)

    def test_write_region_errors(self) -> None:
        data = Dataset({'u': (('x',), np.arange(5))})
        data2 = Dataset({'u': (('x',), np.array([10, 11]))})

        @contextlib.contextmanager
        def setup_and_verify_store(expected=data):
            with self.create_zarr_target() as store:
                data.to_zarr(store, **self.version_kwargs)
                yield store
                with self.open(store) as actual:
                    assert_identical(actual, expected)
        expected = Dataset({'u': (('x',), np.array([10, 11, 2, 3, 4]))})
        with setup_and_verify_store(expected) as store:
            data2.to_zarr(store, region={'x': slice(2)}, **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match=re.escape("cannot set region unless mode='a', mode='a-', mode='r+' or mode=None")):
                data.to_zarr(store, region={'x': slice(None)}, mode='w', **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(TypeError, match='must be a dict'):
                data.to_zarr(store, region=slice(None), **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(TypeError, match='must be slice objects'):
                data2.to_zarr(store, region={'x': [0, 1]}, **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match='step on all slices'):
                data2.to_zarr(store, region={'x': slice(None, None, 2)}, **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match='all keys in ``region`` are not in Dataset dimensions'):
                data.to_zarr(store, region={'y': slice(None)}, **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match='all variables in the dataset to write must have at least one dimension in common'):
                data2.assign(v=2).to_zarr(store, region={'x': slice(2)}, **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match='cannot list the same dimension in both'):
                data.to_zarr(store, region={'x': slice(None)}, append_dim='x', **self.version_kwargs)
        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match="variable 'u' already exists with different dimension sizes"):
                data2.to_zarr(store, region={'x': slice(3)}, **self.version_kwargs)

    @requires_dask
    def test_encoding_chunksizes(self) -> None:
        nx, ny, nt = (4, 4, 5)
        original = xr.Dataset({}, coords={'x': np.arange(nx), 'y': np.arange(ny), 't': np.arange(nt)})
        original['v'] = xr.Variable(('x', 'y', 't'), np.zeros((nx, ny, nt)))
        original = original.chunk({'t': 1, 'x': 2, 'y': 2})
        with self.roundtrip(original) as ds1:
            assert_equal(ds1, original)
            with self.roundtrip(ds1.isel(t=0)) as ds2:
                assert_equal(ds2, original.isel(t=0))

    @requires_dask
    def test_chunk_encoding_with_partial_dask_chunks(self) -> None:
        original = xr.Dataset({'x': xr.DataArray(np.random.random(size=(6, 8)), dims=('a', 'b'))}).chunk({'a': 3})
        with self.roundtrip(original, save_kwargs={'encoding': {'x': {'chunks': [3, 2]}}}) as ds1:
            assert_equal(ds1, original)

    @requires_dask
    def test_chunk_encoding_with_larger_dask_chunks(self) -> None:
        original = xr.Dataset({'a': ('x', [1, 2, 3, 4])}).chunk({'x': 2})
        with self.roundtrip(original, save_kwargs={'encoding': {'a': {'chunks': [1]}}}) as ds1:
            assert_equal(ds1, original)

    @requires_cftime
    def test_open_zarr_use_cftime(self) -> None:
        ds = create_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, **self.version_kwargs)
            ds_a = xr.open_zarr(store_target, **self.version_kwargs)
            assert_identical(ds, ds_a)
            ds_b = xr.open_zarr(store_target, use_cftime=True, **self.version_kwargs)
            assert xr.coding.times.contains_cftime_datetimes(ds_b.time.variable)

    def test_write_read_select_write(self) -> None:
        ds = create_test_data()
        with self.create_zarr_target() as initial_store:
            ds.to_zarr(initial_store, mode='w', **self.version_kwargs)
            ds1 = xr.open_zarr(initial_store, **self.version_kwargs)
            ds_sel = ds1.where(ds1.coords['dim3'] == 'a', drop=True).squeeze('dim3')
            with self.create_zarr_target() as final_store:
                ds_sel.to_zarr(final_store, mode='w', **self.version_kwargs)

    @pytest.mark.parametrize('obj', [Dataset(), DataArray(name='foo')])
    def test_attributes(self, obj) -> None:
        obj = obj.copy()
        obj.attrs['good'] = {'key': 'value'}
        ds = obj if isinstance(obj, Dataset) else obj.to_dataset()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, **self.version_kwargs)
            assert_identical(ds, xr.open_zarr(store_target, **self.version_kwargs))
        obj.attrs['bad'] = DataArray()
        ds = obj if isinstance(obj, Dataset) else obj.to_dataset()
        with self.create_zarr_target() as store_target:
            with pytest.raises(TypeError, match='Invalid attribute in Dataset.attrs.'):
                ds.to_zarr(store_target, **self.version_kwargs)

    @requires_dask
    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
    def test_chunked_datetime64_or_timedelta64(self, dtype) -> None:
        original = create_test_data().astype(dtype).chunk(1)
        with self.roundtrip(original, open_kwargs={'chunks': {}}) as actual:
            for name, actual_var in actual.variables.items():
                assert original[name].chunks == actual_var.chunks
            assert original.chunks == actual.chunks

    @requires_cftime
    @requires_dask
    def test_chunked_cftime_datetime(self) -> None:
        times = cftime_range('2000', freq='D', periods=3)
        original = xr.Dataset(data_vars={'chunked_times': (['time'], times)})
        original = original.chunk({'time': 1})
        with self.roundtrip(original, open_kwargs={'chunks': {}}) as actual:
            for name, actual_var in actual.variables.items():
                assert original[name].chunks == actual_var.chunks
            assert original.chunks == actual.chunks

    def test_vectorized_indexing_negative_step(self) -> None:
        if not has_dask:
            pytest.xfail(reason='zarr without dask handles negative steps in slices incorrectly')
        super().test_vectorized_indexing_negative_step()