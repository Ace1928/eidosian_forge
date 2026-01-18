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
class TestDataArrayToZarr:

    def test_dataarray_to_zarr_no_name(self, tmp_store) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)))
        original_da.to_zarr(tmp_store)
        with open_dataarray(tmp_store, engine='zarr') as loaded_da:
            assert_identical(original_da, loaded_da)

    def test_dataarray_to_zarr_with_name(self, tmp_store) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)), name='test')
        original_da.to_zarr(tmp_store)
        with open_dataarray(tmp_store, engine='zarr') as loaded_da:
            assert_identical(original_da, loaded_da)

    def test_dataarray_to_zarr_coord_name_clash(self, tmp_store) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'], name='x')
        original_da.to_zarr(tmp_store)
        with open_dataarray(tmp_store, engine='zarr') as loaded_da:
            assert_identical(original_da, loaded_da)

    def test_open_dataarray_options(self, tmp_store) -> None:
        data = DataArray(np.arange(5), coords={'y': ('x', range(5))}, dims=['x'])
        data.to_zarr(tmp_store)
        expected = data.drop_vars('y')
        with open_dataarray(tmp_store, engine='zarr', drop_variables=['y']) as loaded:
            assert_identical(expected, loaded)

    @requires_dask
    def test_dataarray_to_zarr_compute_false(self, tmp_store) -> None:
        from dask.delayed import Delayed
        original_da = DataArray(np.arange(12).reshape((3, 4)))
        output = original_da.to_zarr(tmp_store, compute=False)
        assert isinstance(output, Delayed)
        output.compute()
        with open_dataarray(tmp_store, engine='zarr') as loaded_da:
            assert_identical(original_da, loaded_da)