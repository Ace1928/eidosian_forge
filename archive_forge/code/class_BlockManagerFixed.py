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
class BlockManagerFixed(GenericFixed):
    attributes = ['ndim', 'nblocks']
    nblocks: int

    @property
    def shape(self) -> Shape | None:
        try:
            ndim = self.ndim
            items = 0
            for i in range(self.nblocks):
                node = getattr(self.group, f'block{i}_items')
                shape = getattr(node, 'shape', None)
                if shape is not None:
                    items += shape[0]
            node = self.group.block0_values
            shape = getattr(node, 'shape', None)
            if shape is not None:
                shape = list(shape[0:ndim - 1])
            else:
                shape = []
            shape.append(items)
            return shape
        except AttributeError:
            return None

    def read(self, where=None, columns=None, start: int | None=None, stop: int | None=None) -> DataFrame:
        self.validate_read(columns, where)
        select_axis = self.obj_type()._get_block_manager_axis(0)
        axes = []
        for i in range(self.ndim):
            _start, _stop = (start, stop) if i == select_axis else (None, None)
            ax = self.read_index(f'axis{i}', start=_start, stop=_stop)
            axes.append(ax)
        items = axes[0]
        dfs = []
        for i in range(self.nblocks):
            blk_items = self.read_index(f'block{i}_items')
            values = self.read_array(f'block{i}_values', start=_start, stop=_stop)
            columns = items[items.get_indexer(blk_items)]
            df = DataFrame(values.T, columns=columns, index=axes[1], copy=False)
            if using_pyarrow_string_dtype() and is_string_array(values, skipna=True):
                df = df.astype('string[pyarrow_numpy]')
            dfs.append(df)
        if len(dfs) > 0:
            out = concat(dfs, axis=1, copy=True)
            if using_copy_on_write():
                out = out.copy()
            out = out.reindex(columns=items, copy=False)
            return out
        return DataFrame(columns=axes[0], index=axes[1])

    def write(self, obj, **kwargs) -> None:
        super().write(obj, **kwargs)
        if isinstance(obj._mgr, ArrayManager):
            obj = obj._as_manager('block')
        data = obj._mgr
        if not data.is_consolidated():
            data = data.consolidate()
        self.attrs.ndim = data.ndim
        for i, ax in enumerate(data.axes):
            if i == 0 and (not ax.is_unique):
                raise ValueError('Columns index has to be unique for fixed format')
            self.write_index(f'axis{i}', ax)
        self.attrs.nblocks = len(data.blocks)
        for i, blk in enumerate(data.blocks):
            blk_items = data.items.take(blk.mgr_locs)
            self.write_array(f'block{i}_values', blk.values, items=blk_items)
            self.write_index(f'block{i}_items', blk_items)