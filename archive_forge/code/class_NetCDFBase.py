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
class NetCDFBase(CFEncodedBase):
    """Tests for all netCDF3 and netCDF4 backends."""

    @pytest.mark.skipif(ON_WINDOWS, reason='Windows does not allow modifying open files')
    def test_refresh_from_disk(self) -> None:
        with create_tmp_file() as example_1_path:
            with create_tmp_file() as example_1_modified_path:
                with open_example_dataset('example_1.nc') as example_1:
                    self.save(example_1, example_1_path)
                    example_1.rh.values += 100
                    self.save(example_1, example_1_modified_path)
                a = open_dataset(example_1_path, engine=self.engine).load()
                shutil.copy(example_1_modified_path, example_1_path)
                b = open_dataset(example_1_path, engine=self.engine).load()
                try:
                    assert not np.array_equal(a.rh.values, b.rh.values)
                finally:
                    a.close()
                    b.close()