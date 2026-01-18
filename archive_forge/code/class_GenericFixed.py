from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
class GenericFixed(Fixed):
    """a generified fixed version"""
    _index_type_map = {DatetimeIndex: 'datetime', PeriodIndex: 'period'}
    _reverse_index_map = {v: k for k, v in _index_type_map.items()}
    attributes: list[str] = []

    def _class_to_alias(self, cls) -> str:
        return self._index_type_map.get(cls, '')

    def _alias_to_class(self, alias):
        if isinstance(alias, type):
            return alias
        return self._reverse_index_map.get(alias, Index)

    def _get_index_factory(self, attrs):
        index_class = self._alias_to_class(_ensure_decoded(getattr(attrs, 'index_class', '')))
        factory: Callable
        if index_class == DatetimeIndex:

            def f(values, freq=None, tz=None):
                dta = DatetimeArray._simple_new(values.values, dtype=values.dtype, freq=freq)
                result = DatetimeIndex._simple_new(dta, name=None)
                if tz is not None:
                    result = result.tz_localize('UTC').tz_convert(tz)
                return result
            factory = f
        elif index_class == PeriodIndex:

            def f(values, freq=None, tz=None):
                dtype = PeriodDtype(freq)
                parr = PeriodArray._simple_new(values, dtype=dtype)
                return PeriodIndex._simple_new(parr, name=None)
            factory = f
        else:
            factory = index_class
        kwargs = {}
        if 'freq' in attrs:
            kwargs['freq'] = attrs['freq']
            if index_class is Index:
                factory = TimedeltaIndex
        if 'tz' in attrs:
            if isinstance(attrs['tz'], bytes):
                kwargs['tz'] = attrs['tz'].decode('utf-8')
            else:
                kwargs['tz'] = attrs['tz']
            assert index_class is DatetimeIndex
        return (factory, kwargs)

    def validate_read(self, columns, where) -> None:
        """
        raise if any keywords are passed which are not-None
        """
        if columns is not None:
            raise TypeError('cannot pass a column specification when reading a Fixed format store. this store must be selected in its entirety')
        if where is not None:
            raise TypeError('cannot pass a where specification when reading from a Fixed format store. this store must be selected in its entirety')

    @property
    def is_exists(self) -> bool:
        return True

    def set_attrs(self) -> None:
        """set our object attributes"""
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def get_attrs(self) -> None:
        """retrieve our attributes"""
        self.encoding = _ensure_encoding(getattr(self.attrs, 'encoding', None))
        self.errors = _ensure_decoded(getattr(self.attrs, 'errors', 'strict'))
        for n in self.attributes:
            setattr(self, n, _ensure_decoded(getattr(self.attrs, n, None)))

    def write(self, obj, **kwargs) -> None:
        self.set_attrs()

    def read_array(self, key: str, start: int | None=None, stop: int | None=None):
        """read an array for the specified node (off of group"""
        import tables
        node = getattr(self.group, key)
        attrs = node._v_attrs
        transposed = getattr(attrs, 'transposed', False)
        if isinstance(node, tables.VLArray):
            ret = node[0][start:stop]
        else:
            dtype = _ensure_decoded(getattr(attrs, 'value_type', None))
            shape = getattr(attrs, 'shape', None)
            if shape is not None:
                ret = np.empty(shape, dtype=dtype)
            else:
                ret = node[start:stop]
            if dtype and dtype.startswith('datetime64'):
                tz = getattr(attrs, 'tz', None)
                ret = _set_tz(ret, tz, coerce=True)
            elif dtype == 'timedelta64':
                ret = np.asarray(ret, dtype='m8[ns]')
        if transposed:
            return ret.T
        else:
            return ret

    def read_index(self, key: str, start: int | None=None, stop: int | None=None) -> Index:
        variety = _ensure_decoded(getattr(self.attrs, f'{key}_variety'))
        if variety == 'multi':
            return self.read_multi_index(key, start=start, stop=stop)
        elif variety == 'regular':
            node = getattr(self.group, key)
            index = self.read_index_node(node, start=start, stop=stop)
            return index
        else:
            raise TypeError(f'unrecognized index variety: {variety}')

    def write_index(self, key: str, index: Index) -> None:
        if isinstance(index, MultiIndex):
            setattr(self.attrs, f'{key}_variety', 'multi')
            self.write_multi_index(key, index)
        else:
            setattr(self.attrs, f'{key}_variety', 'regular')
            converted = _convert_index('index', index, self.encoding, self.errors)
            self.write_array(key, converted.values)
            node = getattr(self.group, key)
            node._v_attrs.kind = converted.kind
            node._v_attrs.name = index.name
            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                node._v_attrs.index_class = self._class_to_alias(type(index))
            if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
                node._v_attrs.freq = index.freq
            if isinstance(index, DatetimeIndex) and index.tz is not None:
                node._v_attrs.tz = _get_tz(index.tz)

    def write_multi_index(self, key: str, index: MultiIndex) -> None:
        setattr(self.attrs, f'{key}_nlevels', index.nlevels)
        for i, (lev, level_codes, name) in enumerate(zip(index.levels, index.codes, index.names)):
            if isinstance(lev.dtype, ExtensionDtype):
                raise NotImplementedError('Saving a MultiIndex with an extension dtype is not supported.')
            level_key = f'{key}_level{i}'
            conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
            self.write_array(level_key, conv_level.values)
            node = getattr(self.group, level_key)
            node._v_attrs.kind = conv_level.kind
            node._v_attrs.name = name
            setattr(node._v_attrs, f'{key}_name{name}', name)
            label_key = f'{key}_label{i}'
            self.write_array(label_key, level_codes)

    def read_multi_index(self, key: str, start: int | None=None, stop: int | None=None) -> MultiIndex:
        nlevels = getattr(self.attrs, f'{key}_nlevels')
        levels = []
        codes = []
        names: list[Hashable] = []
        for i in range(nlevels):
            level_key = f'{key}_level{i}'
            node = getattr(self.group, level_key)
            lev = self.read_index_node(node, start=start, stop=stop)
            levels.append(lev)
            names.append(lev.name)
            label_key = f'{key}_label{i}'
            level_codes = self.read_array(label_key, start=start, stop=stop)
            codes.append(level_codes)
        return MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=True)

    def read_index_node(self, node: Node, start: int | None=None, stop: int | None=None) -> Index:
        data = node[start:stop]
        if 'shape' in node._v_attrs and np.prod(node._v_attrs.shape) == 0:
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        kind = _ensure_decoded(node._v_attrs.kind)
        name = None
        if 'name' in node._v_attrs:
            name = _ensure_str(node._v_attrs.name)
            name = _ensure_decoded(name)
        attrs = node._v_attrs
        factory, kwargs = self._get_index_factory(attrs)
        if kind in ('date', 'object'):
            index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), dtype=object, **kwargs)
        else:
            index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), **kwargs)
        index.name = name
        return index

    def write_array_empty(self, key: str, value: ArrayLike) -> None:
        """write a 0-len array"""
        arr = np.empty((1,) * value.ndim)
        self._handle.create_array(self.group, key, arr)
        node = getattr(self.group, key)
        node._v_attrs.value_type = str(value.dtype)
        node._v_attrs.shape = value.shape

    def write_array(self, key: str, obj: AnyArrayLike, items: Index | None=None) -> None:
        value = extract_array(obj, extract_numpy=True)
        if key in self.group:
            self._handle.remove_node(self.group, key)
        empty_array = value.size == 0
        transposed = False
        if isinstance(value.dtype, CategoricalDtype):
            raise NotImplementedError('Cannot store a category dtype in a HDF5 dataset that uses format="fixed". Use format="table".')
        if not empty_array:
            if hasattr(value, 'T'):
                value = value.T
                transposed = True
        atom = None
        if self._filters is not None:
            with suppress(ValueError):
                atom = _tables().Atom.from_dtype(value.dtype)
        if atom is not None:
            if not empty_array:
                ca = self._handle.create_carray(self.group, key, atom, value.shape, filters=self._filters)
                ca[:] = value
            else:
                self.write_array_empty(key, value)
        elif value.dtype.type == np.object_:
            inferred_type = lib.infer_dtype(value, skipna=False)
            if empty_array:
                pass
            elif inferred_type == 'string':
                pass
            else:
                ws = performance_doc % (inferred_type, key, items)
                warnings.warn(ws, PerformanceWarning, stacklevel=find_stack_level())
            vlarr = self._handle.create_vlarray(self.group, key, _tables().ObjectAtom())
            vlarr.append(value)
        elif lib.is_np_dtype(value.dtype, 'M'):
            self._handle.create_array(self.group, key, value.view('i8'))
            getattr(self.group, key)._v_attrs.value_type = str(value.dtype)
        elif isinstance(value.dtype, DatetimeTZDtype):
            self._handle.create_array(self.group, key, value.asi8)
            node = getattr(self.group, key)
            node._v_attrs.tz = _get_tz(value.tz)
            node._v_attrs.value_type = f'datetime64[{value.dtype.unit}]'
        elif lib.is_np_dtype(value.dtype, 'm'):
            self._handle.create_array(self.group, key, value.view('i8'))
            getattr(self.group, key)._v_attrs.value_type = 'timedelta64'
        elif empty_array:
            self.write_array_empty(key, value)
        else:
            self._handle.create_array(self.group, key, value)
        getattr(self.group, key)._v_attrs.transposed = transposed