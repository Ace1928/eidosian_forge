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
class IndexCol:
    """
    an index column description class

    Parameters
    ----------
    axis   : axis which I reference
    values : the ndarray like converted values
    kind   : a string description of this type
    typ    : the pytables type
    pos    : the position in the pytables

    """
    is_an_indexable: bool = True
    is_data_indexable: bool = True
    _info_fields = ['freq', 'tz', 'index_name']

    def __init__(self, name: str, values=None, kind=None, typ=None, cname: str | None=None, axis=None, pos=None, freq=None, tz=None, index_name=None, ordered=None, table=None, meta=None, metadata=None) -> None:
        if not isinstance(name, str):
            raise ValueError('`name` must be a str.')
        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata
        if pos is not None:
            self.set_pos(pos)
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self) -> int:
        return self.typ.itemsize

    @property
    def kind_attr(self) -> str:
        return f'{self.name}_kind'

    def set_pos(self, pos: int) -> None:
        """set the position of this column in the Table"""
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'axis', 'pos', 'kind'], temp)])

    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
        return all((getattr(self, a, None) == getattr(other, a, None) for a in ['name', 'cname', 'axis', 'pos']))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    @property
    def is_indexed(self) -> bool:
        """return whether I am an indexed column"""
        if not hasattr(self.table, 'cols'):
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str) -> tuple[np.ndarray, np.ndarray] | tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname].copy()
        val_kind = _ensure_decoded(self.kind)
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs = {}
        kwargs['name'] = _ensure_decoded(self.index_name)
        if self.freq is not None:
            kwargs['freq'] = _ensure_decoded(self.freq)
        factory: type[Index | DatetimeIndex] = Index
        if lib.is_np_dtype(values.dtype, 'M') or isinstance(values.dtype, DatetimeTZDtype):
            factory = DatetimeIndex
        elif values.dtype == 'i8' and 'freq' in kwargs:
            factory = lambda x, **kwds: PeriodIndex.from_ordinals(x, freq=kwds.get('freq', None))._rename(kwds['name'])
        try:
            new_pd_index = factory(values, **kwargs)
        except ValueError:
            if 'freq' in kwargs:
                kwargs['freq'] = None
            new_pd_index = factory(values, **kwargs)
        final_pd_index = _set_tz(new_pd_index, self.tz)
        return (final_pd_index, final_pd_index)

    def take_data(self):
        """return the values"""
        return self.values

    @property
    def attrs(self):
        return self.table._v_attrs

    @property
    def description(self):
        return self.table.description

    @property
    def col(self):
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self):
        """return my cython values"""
        return self.values

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def maybe_set_size(self, min_itemsize=None) -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        if _ensure_decoded(self.kind) == 'string':
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    def validate_names(self) -> None:
        pass

    def validate_and_set(self, handler: AppendableTable, append: bool) -> None:
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize=None):
        """validate this column: return the compared against itemsize"""
        if _ensure_decoded(self.kind) == 'string':
            c = self.col
            if c is not None:
                if itemsize is None:
                    itemsize = self.itemsize
                if c.itemsize < itemsize:
                    raise ValueError(f'Trying to store a string with len [{itemsize}] in [{self.cname}] column but\nthis column has a limit of [{c.itemsize}]!\nConsider using min_itemsize to preset the sizes on these columns')
                return c.itemsize
        return None

    def validate_attr(self, append: bool) -> None:
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(f'incompatible kind in col [{existing_kind} - {self.kind}]')

    def update_info(self, info) -> None:
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})
            existing_value = idx.get(key)
            if key in idx and value is not None and (existing_value != value):
                if key in ['freq', 'index_name']:
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    warnings.warn(ws, AttributeConflictWarning, stacklevel=find_stack_level())
                    idx[key] = None
                    setattr(self, key, None)
                else:
                    raise ValueError(f'invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]')
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info) -> None:
        """set my state from the passed info"""
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler: AppendableTable) -> None:
        """validate that kind=category does not change the categories"""
        if self.meta == 'category':
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if new_metadata is not None and cur_metadata is not None and (not array_equivalent(new_metadata, cur_metadata, strict_nan=True, dtype_equal=True)):
                raise ValueError('cannot append a categorical with different categories to the existing')

    def write_metadata(self, handler: AppendableTable) -> None:
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)