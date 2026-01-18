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
class AppendableTable(Table):
    """support the new appendable table formats"""
    table_type = 'appendable'

    def write(self, obj, axes=None, append: bool=False, complib=None, complevel=None, fletcher32=None, min_itemsize=None, chunksize: int | None=None, expectedrows=None, dropna: bool=False, nan_rep=None, data_columns=None, track_times: bool=True) -> None:
        if not append and self.is_exists:
            self._handle.remove_node(self.group, 'table')
        table = self._create_axes(axes=axes, obj=obj, validate=append, min_itemsize=min_itemsize, nan_rep=nan_rep, data_columns=data_columns)
        for a in table.axes:
            a.validate_names()
        if not table.is_exists:
            options = table.create_description(complib=complib, complevel=complevel, fletcher32=fletcher32, expectedrows=expectedrows)
            table.set_attrs()
            options['track_times'] = track_times
            table._handle.create_table(table.group, **options)
        table.attrs.info = table.info
        for a in table.axes:
            a.validate_and_set(table, append)
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize: int | None, dropna: bool=False) -> None:
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        names = self.dtype.names
        nrows = self.nrows_expected
        masks = []
        if dropna:
            for a in self.values_axes:
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype('u1', copy=False))
        if len(masks):
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
        bvalues = []
        for i, v in enumerate(values):
            new_shape = (nrows,) + self.dtype[names[nindexes + i]].shape
            bvalues.append(v.reshape(new_shape))
        if chunksize is None:
            chunksize = 100000
        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break
            self.write_data_chunk(rows, indexes=[a[start_i:end_i] for a in indexes], mask=mask[start_i:end_i] if mask is not None else None, values=[v[start_i:end_i] for v in bvalues])

    def write_data_chunk(self, rows: np.ndarray, indexes: list[np.ndarray], mask: npt.NDArray[np.bool_] | None, values: list[np.ndarray]) -> None:
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
        for v in values:
            if not np.prod(v.shape):
                return
        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]
        if len(rows):
            self.table.append(rows)
            self.table.flush()

    def delete(self, where=None, start: int | None=None, stop: int | None=None):
        if where is None or not len(where):
            if start is None and stop is None:
                nrows = self.nrows
                self._handle.remove_node(self.group, recursive=True)
            else:
                if stop is None:
                    stop = self.nrows
                nrows = self.table.remove_rows(start=start, stop=stop)
                self.table.flush()
            return nrows
        if not self.infer_axes():
            return None
        table = self.table
        selection = Selection(self, where, start=start, stop=stop)
        values = selection.select_coords()
        sorted_series = Series(values, copy=False).sort_values()
        ln = len(sorted_series)
        if ln:
            diff = sorted_series.diff()
            groups = list(diff[diff > 1].index)
            if not len(groups):
                groups = [0]
            if groups[-1] != ln:
                groups.append(ln)
            if groups[0] != 0:
                groups.insert(0, 0)
            pg = groups.pop()
            for g in reversed(groups):
                rows = sorted_series.take(range(g, pg))
                table.remove_rows(start=rows[rows.index[0]], stop=rows[rows.index[-1]] + 1)
                pg = g
            self.table.flush()
        return ln