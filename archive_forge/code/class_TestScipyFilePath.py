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
@requires_scipy
class TestScipyFilePath(CFEncodedBase, NetCDF3Only):
    engine: T_NetcdfEngine = 'scipy'

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.ScipyDataStore(tmp_file, mode='w') as store:
                yield store

    def test_array_attrs(self) -> None:
        ds = Dataset(attrs={'foo': [[1, 2], [3, 4]]})
        with pytest.raises(ValueError, match='must be 1-dimensional'):
            with self.roundtrip(ds):
                pass

    def test_roundtrip_example_1_netcdf_gz(self) -> None:
        with open_example_dataset('example_1.nc.gz') as expected:
            with open_example_dataset('example_1.nc') as actual:
                assert_identical(expected, actual)

    def test_netcdf3_endianness(self) -> None:
        with open_example_dataset('bears.nc', engine='scipy') as expected:
            for var in expected.variables.values():
                assert var.dtype.isnative

    @requires_netCDF4
    def test_nc4_scipy(self) -> None:
        with create_tmp_file(allow_cleanup_failure=True) as tmp_file:
            with nc4.Dataset(tmp_file, 'w', format='NETCDF4') as rootgrp:
                rootgrp.createGroup('foo')
            with pytest.raises(TypeError, match='pip install netcdf4'):
                open_dataset(tmp_file, engine='scipy')