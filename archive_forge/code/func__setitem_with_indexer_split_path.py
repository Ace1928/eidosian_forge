from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
def _setitem_with_indexer_split_path(self, indexer, value, name: str):
    """
        Setitem column-wise.
        """
    assert self.ndim == 2
    if not isinstance(indexer, tuple):
        indexer = _tuplify(self.ndim, indexer)
    if len(indexer) > self.ndim:
        raise IndexError('too many indices for array')
    if isinstance(indexer[0], np.ndarray) and indexer[0].ndim > 2:
        raise ValueError('Cannot set values with ndim > 2')
    if isinstance(value, ABCSeries) and name != 'iloc' or isinstance(value, dict):
        from pandas import Series
        value = self._align_series(indexer, Series(value))
    info_axis = indexer[1]
    ilocs = self._ensure_iterable_column_indexer(info_axis)
    pi = indexer[0]
    lplane_indexer = length_of_indexer(pi, self.obj.index)
    if is_list_like_indexer(value) and getattr(value, 'ndim', 1) > 0:
        if isinstance(value, ABCDataFrame):
            self._setitem_with_indexer_frame_value(indexer, value, name)
        elif np.ndim(value) == 2:
            self._setitem_with_indexer_2d_value(indexer, value)
        elif len(ilocs) == 1 and lplane_indexer == len(value) and (not is_scalar(pi)):
            self._setitem_single_column(ilocs[0], value, pi)
        elif len(ilocs) == 1 and 0 != lplane_indexer != len(value):
            if len(value) == 1 and (not is_integer(info_axis)):
                return self._setitem_with_indexer((pi, info_axis[0]), value[0])
            raise ValueError('Must have equal len keys and value when setting with an iterable')
        elif lplane_indexer == 0 and len(value) == len(self.obj.index):
            pass
        elif self._is_scalar_access(indexer) and is_object_dtype(self.obj.dtypes._values[ilocs[0]]):
            self._setitem_single_column(indexer[1], value, pi)
        elif len(ilocs) == len(value):
            for loc, v in zip(ilocs, value):
                self._setitem_single_column(loc, v, pi)
        elif len(ilocs) == 1 and com.is_null_slice(pi) and (len(self.obj) == 0):
            self._setitem_single_column(ilocs[0], value, pi)
        else:
            raise ValueError('Must have equal len keys and value when setting with an iterable')
    else:
        for loc in ilocs:
            self._setitem_single_column(loc, value, pi)