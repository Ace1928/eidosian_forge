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
class TestValidateAttrs:

    def test_validating_attrs(self) -> None:

        def new_dataset():
            return Dataset({'data': ('y', np.arange(10.0))}, {'y': np.arange(10)})

        def new_dataset_and_dataset_attrs():
            ds = new_dataset()
            return (ds, ds.attrs)

        def new_dataset_and_data_attrs():
            ds = new_dataset()
            return (ds, ds.data.attrs)

        def new_dataset_and_coord_attrs():
            ds = new_dataset()
            return (ds, ds.coords['y'].attrs)
        for new_dataset_and_attrs in [new_dataset_and_dataset_attrs, new_dataset_and_data_attrs, new_dataset_and_coord_attrs]:
            ds, attrs = new_dataset_and_attrs()
            attrs[123] = 'test'
            with pytest.raises(TypeError, match='Invalid name for attr: 123'):
                ds.to_netcdf('test.nc')
            ds, attrs = new_dataset_and_attrs()
            attrs[MiscObject()] = 'test'
            with pytest.raises(TypeError, match='Invalid name for attr: '):
                ds.to_netcdf('test.nc')
            ds, attrs = new_dataset_and_attrs()
            attrs[''] = 'test'
            with pytest.raises(ValueError, match="Invalid name for attr '':"):
                ds.to_netcdf('test.nc')
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 'test'
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = {'a': 5}
            with pytest.raises(TypeError, match="Invalid value for attr 'test'"):
                ds.to_netcdf('test.nc')
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = MiscObject()
            with pytest.raises(TypeError, match="Invalid value for attr 'test'"):
                ds.to_netcdf('test.nc')
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 5
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 3.14
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = [1, 2, 3, 4]
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = (1.9, 2.5)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = np.arange(5)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = 'This is a string'
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)
            ds, attrs = new_dataset_and_attrs()
            attrs['test'] = ''
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)