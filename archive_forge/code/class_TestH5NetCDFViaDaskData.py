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
@requires_dask
@pytest.mark.filterwarnings('ignore:deallocating CachingFileManager')
class TestH5NetCDFViaDaskData(TestH5NetCDFData):

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        open_kwargs.setdefault('chunks', -1)
        with TestH5NetCDFData.roundtrip(self, data, save_kwargs, open_kwargs, allow_cleanup_failure) as ds:
            yield ds

    @pytest.mark.skip(reason='caching behavior differs for dask')
    def test_dataset_caching(self) -> None:
        pass

    def test_write_inconsistent_chunks(self) -> None:
        x = da.zeros((100, 100), dtype='f4', chunks=(50, 100))
        x = DataArray(data=x, dims=('lat', 'lon'), name='x')
        x.encoding['chunksizes'] = (50, 100)
        x.encoding['original_shape'] = (100, 100)
        y = da.ones((100, 100), dtype='f4', chunks=(100, 50))
        y = DataArray(data=y, dims=('lat', 'lon'), name='y')
        y.encoding['chunksizes'] = (100, 50)
        y.encoding['original_shape'] = (100, 100)
        ds = Dataset({'x': x, 'y': y})
        with self.roundtrip(ds) as actual:
            assert actual['x'].encoding['chunksizes'] == (50, 100)
            assert actual['y'].encoding['chunksizes'] == (100, 50)