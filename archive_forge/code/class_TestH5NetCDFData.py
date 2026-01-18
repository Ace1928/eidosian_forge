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
@requires_h5netcdf
@requires_netCDF4
@pytest.mark.filterwarnings('ignore:use make_scale(name) instead')
class TestH5NetCDFData(NetCDF4Base):
    engine: T_NetcdfEngine = 'h5netcdf'

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.H5NetCDFStore.open(tmp_file, 'w')

    def test_complex(self) -> None:
        expected = Dataset({'x': ('y', np.ones(5) + 1j * np.ones(5))})
        save_kwargs = {'invalid_netcdf': True}
        with pytest.warns(UserWarning, match='You are writing invalid netcdf features'):
            with self.roundtrip(expected, save_kwargs=save_kwargs) as actual:
                assert_equal(expected, actual)

    @pytest.mark.parametrize('invalid_netcdf', [None, False])
    def test_complex_error(self, invalid_netcdf) -> None:
        import h5netcdf
        expected = Dataset({'x': ('y', np.ones(5) + 1j * np.ones(5))})
        save_kwargs = {'invalid_netcdf': invalid_netcdf}
        with pytest.raises(h5netcdf.CompatibilityError, match='are not a supported NetCDF feature'):
            with self.roundtrip(expected, save_kwargs=save_kwargs) as actual:
                assert_equal(expected, actual)

    def test_numpy_bool_(self) -> None:
        expected = Dataset({'x': ('y', np.ones(5), {'numpy_bool': np.bool_(True)})})
        save_kwargs = {'invalid_netcdf': True}
        with pytest.warns(UserWarning, match='You are writing invalid netcdf features'):
            with self.roundtrip(expected, save_kwargs=save_kwargs) as actual:
                assert_identical(expected, actual)

    def test_cross_engine_read_write_netcdf4(self) -> None:
        data = create_test_data().drop_vars('dim3')
        data.attrs['foo'] = 'bar'
        valid_engines: list[T_NetcdfEngine] = ['netcdf4', 'h5netcdf']
        for write_engine in valid_engines:
            with create_tmp_file() as tmp_file:
                data.to_netcdf(tmp_file, engine=write_engine)
                for read_engine in valid_engines:
                    with open_dataset(tmp_file, engine=read_engine) as actual:
                        assert_identical(data, actual)

    def test_read_byte_attrs_as_unicode(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as nc:
                nc.foo = b'bar'
            with open_dataset(tmp_file) as actual:
                expected = Dataset(attrs={'foo': 'bar'})
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

    def test_compression_encoding_h5py(self) -> None:
        ENCODINGS: tuple[tuple[dict[str, Any], dict[str, Any]], ...] = (({'compression': 'gzip', 'compression_opts': 9}, {'zlib': True, 'complevel': 9}), ({'compression': 'lzf', 'compression_opts': None}, {'compression': 'lzf', 'compression_opts': None}), ({'compression': 'lzf', 'compression_opts': None, 'zlib': True, 'complevel': 9}, {'compression': 'lzf', 'compression_opts': None}))
        for compr_in, compr_out in ENCODINGS:
            data = create_test_data()
            compr_common = {'chunksizes': (5, 5), 'fletcher32': True, 'shuffle': True, 'original_shape': data.var2.shape}
            data['var2'].encoding.update(compr_in)
            data['var2'].encoding.update(compr_common)
            compr_out.update(compr_common)
            data['scalar'] = ('scalar_dim', np.array([2.0]))
            data['scalar'] = data['scalar'][0]
            with self.roundtrip(data) as actual:
                for k, v in compr_out.items():
                    assert v == actual['var2'].encoding[k]

    def test_compression_check_encoding_h5py(self) -> None:
        """When mismatched h5py and NetCDF4-Python encodings are expressed
        in to_netcdf(encoding=...), must raise ValueError
        """
        data = Dataset({'x': ('y', np.arange(10.0))})
        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file, engine='h5netcdf', encoding={'x': {'compression': 'gzip', 'zlib': True, 'compression_opts': 6, 'complevel': 6}})
            with open_dataset(tmp_file, engine='h5netcdf') as actual:
                assert actual.x.encoding['zlib'] is True
                assert actual.x.encoding['complevel'] == 6
        with create_tmp_file() as tmp_file:
            with pytest.raises(ValueError, match="'zlib' and 'compression' encodings mismatch"):
                data.to_netcdf(tmp_file, engine='h5netcdf', encoding={'x': {'compression': 'lzf', 'zlib': True}})
        with create_tmp_file() as tmp_file:
            with pytest.raises(ValueError, match="'complevel' and 'compression_opts' encodings mismatch"):
                data.to_netcdf(tmp_file, engine='h5netcdf', encoding={'x': {'compression': 'gzip', 'compression_opts': 5, 'complevel': 6}})

    def test_dump_encodings_h5py(self) -> None:
        ds = Dataset({'x': ('y', np.arange(10.0))})
        kwargs = {'encoding': {'x': {'compression': 'gzip', 'compression_opts': 9}}}
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding['zlib']
            assert actual.x.encoding['complevel'] == 9
        kwargs = {'encoding': {'x': {'compression': 'lzf', 'compression_opts': None}}}
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding['compression'] == 'lzf'
            assert actual.x.encoding['compression_opts'] is None

    def test_decode_utf8_warning(self) -> None:
        title = b'\xc3'
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, 'w') as f:
                f.title = title
            with pytest.warns(UnicodeWarning, match='returning bytes undecoded') as w:
                ds = xr.load_dataset(tmp_file, engine='h5netcdf')
                assert ds.title == title
                assert "attribute 'title' of h5netcdf object '/'" in str(w[0].message)