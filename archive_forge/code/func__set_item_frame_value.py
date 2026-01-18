from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
def _set_item_frame_value(self, key, value: DataFrame) -> None:
    self._ensure_valid_index(value)
    if key in self.columns:
        loc = self.columns.get_loc(key)
        cols = self.columns[loc]
        len_cols = 1 if is_scalar(cols) or isinstance(cols, tuple) else len(cols)
        if len_cols != len(value.columns):
            raise ValueError('Columns must be same length as key')
        if isinstance(self.columns, MultiIndex) and isinstance(loc, (slice, Series, np.ndarray, Index)):
            cols_droplevel = maybe_droplevels(cols, key)
            if len(cols_droplevel) and (not cols_droplevel.equals(value.columns)):
                value = value.reindex(cols_droplevel, axis=1)
            for col, col_droplevel in zip(cols, cols_droplevel):
                self[col] = value[col_droplevel]
            return
        if is_scalar(cols):
            self[cols] = value[value.columns[0]]
            return
        locs: np.ndarray | list
        if isinstance(loc, slice):
            locs = np.arange(loc.start, loc.stop, loc.step)
        elif is_scalar(loc):
            locs = [loc]
        else:
            locs = loc.nonzero()[0]
        return self.isetitem(locs, value)
    if len(value.columns) > 1:
        raise ValueError(f'Cannot set a DataFrame with multiple columns to the single column {key}')
    elif len(value.columns) == 0:
        raise ValueError(f'Cannot set a DataFrame without columns to the column {key}')
    self[key] = value[value.columns[0]]