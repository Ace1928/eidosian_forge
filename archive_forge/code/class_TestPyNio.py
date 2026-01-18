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
@requires_pynio
class TestPyNio(CFEncodedBase, NetCDF3Only):

    def test_write_store(self) -> None:
        pass

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with open_dataset(path, engine='pynio', **kwargs) as ds:
            yield ds

    def test_kwargs(self) -> None:
        kwargs = {'format': 'grib'}
        path = os.path.join(os.path.dirname(__file__), 'data', 'example')
        with backends.NioDataStore(path, **kwargs) as store:
            assert store._manager._kwargs['format'] == 'grib'

    def save(self, dataset, path, **kwargs):
        return dataset.to_netcdf(path, engine='scipy', **kwargs)

    def test_weakrefs(self) -> None:
        example = Dataset({'foo': ('x', np.arange(5.0))})
        expected = example.rename({'foo': 'bar', 'x': 'y'})
        with create_tmp_file() as tmp_file:
            example.to_netcdf(tmp_file, engine='scipy')
            on_disk = open_dataset(tmp_file, engine='pynio')
            actual = on_disk.rename({'foo': 'bar', 'x': 'y'})
            del on_disk
            assert_identical(actual, expected)