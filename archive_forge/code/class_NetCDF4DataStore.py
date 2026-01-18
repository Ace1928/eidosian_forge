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
class NetCDF4DataStore(WritableCFDataStore):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    __slots__ = ('autoclose', 'format', 'is_remote', 'lock', '_filename', '_group', '_manager', '_mode')

    def __init__(self, manager, group=None, mode=None, lock=NETCDF4_PYTHON_LOCK, autoclose=False):
        import netCDF4
        if isinstance(manager, netCDF4.Dataset):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not netCDF4.Dataset:
                    raise ValueError('must supply a root netCDF4.Dataset if the group argument is provided')
                root = manager
            manager = DummyFileManager(root)
        self._manager = manager
        self._group = group
        self._mode = mode
        self.format = self.ds.data_model
        self._filename = self.ds.filepath()
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self.autoclose = autoclose

    @classmethod
    def open(cls, filename, mode='r', format='NETCDF4', group=None, clobber=True, diskless=False, persist=False, lock=None, lock_maker=None, autoclose=False):
        import netCDF4
        if isinstance(filename, os.PathLike):
            filename = os.fspath(filename)
        if not isinstance(filename, str):
            raise ValueError("can only read bytes or file-like objects with engine='scipy' or 'h5netcdf'")
        if format is None:
            format = 'NETCDF4'
        if lock is None:
            if mode == 'r':
                if is_remote_uri(filename):
                    lock = NETCDFC_LOCK
                else:
                    lock = NETCDF4_PYTHON_LOCK
            else:
                if format is None or format.startswith('NETCDF4'):
                    base_lock = NETCDF4_PYTHON_LOCK
                else:
                    base_lock = NETCDFC_LOCK
                lock = combine_locks([base_lock, get_write_lock(filename)])
        kwargs = dict(clobber=clobber, diskless=diskless, persist=persist, format=format)
        manager = CachingFileManager(netCDF4.Dataset, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = _nc4_require_group(root, self._group, self._mode)
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name: str, var):
        import netCDF4
        dimensions = var.dimensions
        attributes = {k: var.getncattr(k) for k in var.ncattrs()}
        data = indexing.LazilyIndexedArray(NetCDF4ArrayWrapper(name, self))
        encoding: dict[str, Any] = {}
        if isinstance(var.datatype, netCDF4.EnumType):
            encoding['dtype'] = np.dtype(data.dtype, metadata={'enum': var.datatype.enum_dict, 'enum_name': var.datatype.name})
        else:
            encoding['dtype'] = var.dtype
        _ensure_fill_value_valid(data, attributes)
        filters = var.filters()
        if filters is not None:
            encoding.update(filters)
        chunking = var.chunking()
        if chunking is not None:
            if chunking == 'contiguous':
                encoding['contiguous'] = True
                encoding['chunksizes'] = None
            else:
                encoding['contiguous'] = False
                encoding['chunksizes'] = tuple(chunking)
                encoding['preferred_chunks'] = dict(zip(var.dimensions, chunking))
        pop_to(attributes, encoding, 'least_significant_digit')
        encoding['source'] = self._filename
        encoding['original_shape'] = var.shape
        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenDict(((k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()))

    def get_attrs(self):
        return FrozenDict(((k, self.ds.getncattr(k)) for k in self.ds.ncattrs()))

    def get_dimensions(self):
        return FrozenDict(((k, len(v)) for k, v in self.ds.dimensions.items()))

    def get_encoding(self):
        return {'unlimited_dims': {k for k, v in self.ds.dimensions.items() if v.isunlimited()}}

    def set_dimension(self, name, length, is_unlimited=False):
        _ensure_no_forward_slash_in_name(name)
        dim_length = length if not is_unlimited else None
        self.ds.createDimension(name, size=dim_length)

    def set_attribute(self, key, value):
        if self.format != 'NETCDF4':
            value = encode_nc3_attr_value(value)
        if _is_list_of_strings(value):
            self.ds.setncattr_string(key, value)
        else:
            self.ds.setncattr(key, value)

    def encode_variable(self, variable):
        variable = _force_native_endianness(variable)
        if self.format == 'NETCDF4':
            variable = _encode_nc4_variable(variable)
        else:
            variable = encode_nc3_variable(variable)
        return variable

    def prepare_variable(self, name, variable: Variable, check_encoding=False, unlimited_dims=None):
        _ensure_no_forward_slash_in_name(name)
        attrs = variable.attrs.copy()
        fill_value = attrs.pop('_FillValue', None)
        datatype = _get_datatype(variable, self.format, raise_on_invalid_encoding=check_encoding)
        if (meta := np.dtype(datatype).metadata) and (e_name := meta.get('enum_name')) and (e_dict := meta.get('enum')):
            datatype = self._build_and_get_enum(name, datatype, e_name, e_dict)
        encoding = _extract_nc4_variable_encoding(variable, raise_on_invalid=check_encoding, unlimited_dims=unlimited_dims)
        if name in self.ds.variables:
            nc4_var = self.ds.variables[name]
        else:
            default_args = dict(varname=name, datatype=datatype, dimensions=variable.dims, zlib=False, complevel=4, shuffle=True, fletcher32=False, contiguous=False, chunksizes=None, endian='native', least_significant_digit=None, fill_value=fill_value)
            default_args.update(encoding)
            default_args.pop('_FillValue', None)
            nc4_var = self.ds.createVariable(**default_args)
        nc4_var.setncatts(attrs)
        target = NetCDF4ArrayWrapper(name, self)
        return (target, variable.data)

    def _build_and_get_enum(self, var_name: str, dtype: np.dtype, enum_name: str, enum_dict: dict[str, int]) -> Any:
        """
        Add or get the netCDF4 Enum based on the dtype in encoding.
        The return type should be ``netCDF4.EnumType``,
        but we avoid importing netCDF4 globally for performances.
        """
        if enum_name not in self.ds.enumtypes:
            return self.ds.createEnumType(dtype, enum_name, enum_dict)
        datatype = self.ds.enumtypes[enum_name]
        if datatype.enum_dict != enum_dict:
            error_msg = f"Cannot save variable `{var_name}` because an enum `{enum_name}` already exists in the Dataset but have a different definition. To fix this error, make sure each variable have a uniquely named enum in their `encoding['dtype'].metadata` or, if they should share the same enum type, make sure the enums are identical."
            raise ValueError(error_msg)
        return datatype

    def sync(self):
        self.ds.sync()

    def close(self, **kwargs):
        self._manager.close(**kwargs)