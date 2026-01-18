from __future__ import annotations
import functools
import operator
import os
from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import (
from xarray.backends.netcdf3 import encode_nc3_attr_value, encode_nc3_variable
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
class BaseNetCDF4Array(BackendArray):
    __slots__ = ('datastore', 'dtype', 'shape', 'variable_name')

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        dtype = array.dtype
        if dtype is str:
            dtype = coding.strings.create_vlen_dtype(str)
        self.dtype = dtype

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_array(needs_lock=False)
            data[key] = value
            if self.datastore.autoclose:
                self.datastore.close(needs_lock=False)

    def get_array(self, needs_lock=True):
        raise NotImplementedError('Virtual Method')