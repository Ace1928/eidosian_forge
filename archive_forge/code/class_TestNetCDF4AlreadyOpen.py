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
class TestNetCDF4AlreadyOpen:

    def test_base_case(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                v = nc.createVariable('x', 'int')
                v[...] = 42
            nc = nc4.Dataset(tmp_file, mode='r')
            store = backends.NetCDF4DataStore(nc)
            with open_dataset(store) as ds:
                expected = Dataset({'x': ((), 42)})
                assert_identical(expected, ds)

    def test_group(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                group = nc.createGroup('g')
                v = group.createVariable('x', 'int')
                v[...] = 42
            nc = nc4.Dataset(tmp_file, mode='r')
            store = backends.NetCDF4DataStore(nc.groups['g'])
            with open_dataset(store) as ds:
                expected = Dataset({'x': ((), 42)})
                assert_identical(expected, ds)
            nc = nc4.Dataset(tmp_file, mode='r')
            store = backends.NetCDF4DataStore(nc, group='g')
            with open_dataset(store) as ds:
                expected = Dataset({'x': ((), 42)})
                assert_identical(expected, ds)
            with nc4.Dataset(tmp_file, mode='r') as nc:
                with pytest.raises(ValueError, match='must supply a root'):
                    backends.NetCDF4DataStore(nc.groups['g'], group='g')

    def test_deepcopy(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode='w') as nc:
                nc.createDimension('x', 10)
                v = nc.createVariable('y', np.int32, ('x',))
                v[:] = np.arange(10)
            h5 = nc4.Dataset(tmp_file, mode='r')
            store = backends.NetCDF4DataStore(h5)
            with open_dataset(store) as ds:
                copied = ds.copy(deep=True)
                expected = Dataset({'y': ('x', np.arange(10))})
                assert_identical(expected, copied)