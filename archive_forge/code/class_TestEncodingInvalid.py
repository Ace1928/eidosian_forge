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
class TestEncodingInvalid:

    def test_extract_nc4_variable_encoding(self) -> None:
        var = xr.Variable(('x',), [1, 2, 3], {}, {'foo': 'bar'})
        with pytest.raises(ValueError, match='unexpected encoding'):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)
        var = xr.Variable(('x',), [1, 2, 3], {}, {'chunking': (2, 1)})
        encoding = _extract_nc4_variable_encoding(var)
        assert {} == encoding
        var = xr.Variable(('x',), [1, 2, 3], {}, {'shuffle': True})
        encoding = _extract_nc4_variable_encoding(var, raise_on_invalid=True)
        assert {'shuffle': True} == encoding
        var = xr.Variable(('x',), [1, 2, 3], {}, {'contiguous': True})
        encoding = _extract_nc4_variable_encoding(var, unlimited_dims=('x',))
        assert {} == encoding

    @requires_netCDF4
    def test_extract_nc4_variable_encoding_netcdf4(self):
        var = xr.Variable(('x',), [1, 2, 3], {}, {'compression': 'szlib'})
        _extract_nc4_variable_encoding(var, backend='netCDF4', raise_on_invalid=True)

    def test_extract_h5nc_encoding(self) -> None:
        var = xr.Variable(('x',), [1, 2, 3], {}, {'least_sigificant_digit': 2})
        with pytest.raises(ValueError, match='unexpected encoding'):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)