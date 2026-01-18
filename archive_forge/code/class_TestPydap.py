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
@requires_scipy_or_netCDF4
@requires_pydap
@pytest.mark.filterwarnings('ignore:The binary mode of fromstring is deprecated')
class TestPydap:

    def convert_to_pydap_dataset(self, original):
        from pydap.model import BaseType, DatasetType, GridType
        ds = DatasetType('bears', **original.attrs)
        for key, var in original.data_vars.items():
            v = GridType(key)
            v[key] = BaseType(key, var.values, dimensions=var.dims, **var.attrs)
            for d in var.dims:
                v[d] = BaseType(d, var[d].values)
            ds[key] = v
        for d in original.coords:
            ds[d] = BaseType(d, original[d].values, dimensions=(d,), **original[d].attrs)
        return ds

    @contextlib.contextmanager
    def create_datasets(self, **kwargs):
        with open_example_dataset('bears.nc') as expected:
            pydap_ds = self.convert_to_pydap_dataset(expected)
            actual = open_dataset(PydapDataStore(pydap_ds))
            expected['bears'] = expected['bears'].astype(str)
            yield (actual, expected)

    def test_cmp_local_file(self) -> None:
        with self.create_datasets() as (actual, expected):
            assert_equal(actual, expected)
            assert 'NC_GLOBAL' not in actual.attrs
            assert 'history' in actual.attrs
            assert actual.attrs.keys() == expected.attrs.keys()
        with self.create_datasets() as (actual, expected):
            assert_equal(actual[{'l': 2}], expected[{'l': 2}])
        with self.create_datasets() as (actual, expected):
            assert_equal(actual.isel(i=0, j=-1), expected.isel(i=0, j=-1))
        with self.create_datasets() as (actual, expected):
            assert_equal(actual.isel(j=slice(1, 2)), expected.isel(j=slice(1, 2)))
        with self.create_datasets() as (actual, expected):
            indexers = {'i': [1, 0, 0], 'j': [1, 2, 0, 1]}
            assert_equal(actual.isel(**indexers), expected.isel(**indexers))
        with self.create_datasets() as (actual, expected):
            indexers2 = {'i': DataArray([0, 1, 0], dims='a'), 'j': DataArray([0, 2, 1], dims='a')}
            assert_equal(actual.isel(**indexers2), expected.isel(**indexers2))

    def test_compatible_to_netcdf(self) -> None:
        with self.create_datasets() as (actual, expected):
            with create_tmp_file() as tmp_file:
                actual.to_netcdf(tmp_file)
                with open_dataset(tmp_file) as actual2:
                    actual2['bears'] = actual2['bears'].astype(str)
                    assert_equal(actual2, expected)

    @requires_dask
    def test_dask(self) -> None:
        with self.create_datasets(chunks={'j': 2}) as (actual, expected):
            assert_equal(actual, expected)