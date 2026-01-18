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
@requires_netCDF4
class TestNetCDF4Data(NetCDF4Base):

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(tmp_file, mode='w') as store:
                yield store

    def test_variable_order(self) -> None:
        ds = Dataset()
        ds['a'] = 1
        ds['z'] = 2
        ds['b'] = 3
        ds.coords['c'] = 4
        with self.roundtrip(ds) as actual:
            assert list(ds.variables) == list(actual.variables)

    def test_unsorted_index_raises(self) -> None:
        random_data = np.random.random(size=(4, 6))
        dim0 = [0, 1, 2, 3]
        dim1 = [0, 2, 1, 3, 5, 4]
        da = xr.DataArray(data=random_data, dims=('dim0', 'dim1'), coords={'dim0': dim0, 'dim1': dim1}, name='randovar')
        ds = da.to_dataset()
        with self.roundtrip(ds) as ondisk:
            inds = np.argsort(dim1)
            ds2 = ondisk.isel(dim1=inds)
            try:
                ds2.randovar.values
            except IndexError as err:
                assert 'first by calling .load' in str(err)

    def test_setncattr_string(self) -> None:
        list_of_strings = ['list', 'of', 'strings']
        one_element_list_of_strings = ['one element']
        one_string = 'one string'
        attrs = {'foo': list_of_strings, 'bar': one_element_list_of_strings, 'baz': one_string}
        ds = Dataset({'x': ('y', [1, 2, 3], attrs)}, attrs=attrs)
        with self.roundtrip(ds) as actual:
            for totest in [actual, actual['x']]:
                assert_array_equal(list_of_strings, totest.attrs['foo'])
                assert_array_equal(one_element_list_of_strings, totest.attrs['bar'])
                assert one_string == totest.attrs['baz']

    @pytest.mark.parametrize('compression', [None, 'zlib', 'szip', 'zstd', 'blosc_lz', 'blosc_lz4', 'blosc_lz4hc', 'blosc_zlib', 'blosc_zstd'])
    @requires_netCDF4_1_6_2_or_above
    @pytest.mark.xfail(ON_WINDOWS, reason='new compression not yet implemented')
    def test_compression_encoding(self, compression: str | None) -> None:
        data = create_test_data(dim_sizes=(20, 80, 10))
        encoding_params: dict[str, Any] = dict(compression=compression, blosc_shuffle=1)
        data['var2'].encoding.update(encoding_params)
        data['var2'].encoding.update({'chunksizes': (20, 40), 'original_shape': data.var2.shape, 'blosc_shuffle': 1, 'fletcher32': False})
        with self.roundtrip(data) as actual:
            expected_encoding = data['var2'].encoding.copy()
            compression = expected_encoding.pop('compression')
            blosc_shuffle = expected_encoding.pop('blosc_shuffle')
            if compression is not None:
                if 'blosc' in compression and blosc_shuffle:
                    expected_encoding['blosc'] = {'compressor': compression, 'shuffle': blosc_shuffle}
                    expected_encoding['shuffle'] = False
                elif compression == 'szip':
                    expected_encoding['szip'] = {'coding': 'nn', 'pixels_per_block': 8}
                    expected_encoding['shuffle'] = False
                else:
                    expected_encoding[compression] = True
                    if compression == 'zstd':
                        expected_encoding['shuffle'] = False
            else:
                expected_encoding['shuffle'] = False
            actual_encoding = actual['var2'].encoding
            assert expected_encoding.items() <= actual_encoding.items()
        if encoding_params['compression'] is not None and 'blosc' not in encoding_params['compression']:
            expected = data.isel(dim1=0)
            with self.roundtrip(expected) as actual:
                assert_equal(expected, actual)

    @pytest.mark.skip(reason='https://github.com/Unidata/netcdf4-python/issues/1195')
    def test_refresh_from_disk(self) -> None:
        super().test_refresh_from_disk()