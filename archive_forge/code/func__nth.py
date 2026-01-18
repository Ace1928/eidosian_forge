from __future__ import annotations
from collections.abc import (
import datetime
from functools import (
import inspect
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config.config import option_context
from pandas._libs import (
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core._numba import executor
from pandas.core.apply import warn_alias_replacement
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
from pandas.core.indexes.api import (
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
def _nth(self, n: PositionalIndexer | tuple, dropna: Literal['any', 'all', None]=None) -> NDFrameT:
    if not dropna:
        mask = self._make_mask_from_positional_indexer(n)
        ids, _, _ = self._grouper.group_info
        mask = mask & (ids != -1)
        out = self._mask_selected_obj(mask)
        return out
    if not is_integer(n):
        raise ValueError('dropna option only supported for an integer argument')
    if dropna not in ['any', 'all']:
        raise ValueError(f"For a DataFrame or Series groupby.nth, dropna must be either None, 'any' or 'all', (was passed {dropna}).")
    n = cast(int, n)
    dropped = self._selected_obj.dropna(how=dropna, axis=self.axis)
    grouper: np.ndarray | Index | ops.BaseGrouper
    if len(dropped) == len(self._selected_obj):
        grouper = self._grouper
    else:
        axis = self._grouper.axis
        grouper = self._grouper.codes_info[axis.isin(dropped.index)]
        if self._grouper.has_dropped_na:
            nulls = grouper == -1
            values = np.where(nulls, NA, grouper)
            grouper = Index(values, dtype='Int64')
    if self.axis == 1:
        grb = dropped.T.groupby(grouper, as_index=self.as_index, sort=self.sort)
    else:
        grb = dropped.groupby(grouper, as_index=self.as_index, sort=self.sort)
    return grb.nth(n)