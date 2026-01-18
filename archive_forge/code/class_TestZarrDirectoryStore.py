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
@pytest.mark.skipif(ON_WINDOWS, reason='Very flaky on Windows CI. Can re-enable assuming it starts consistently passing.')
class TestZarrDirectoryStore(ZarrBase):

    @contextlib.contextmanager
    def create_zarr_target(self):
        with create_tmp_file(suffix='.zarr') as tmp:
            yield tmp

    @contextlib.contextmanager
    def create_store(self):
        with self.create_zarr_target() as store_target:
            group = backends.ZarrStore.open_group(store_target, mode='w')
            if have_zarr_v3:
                assert group.zarr_group.store._store_version == 2
            yield group