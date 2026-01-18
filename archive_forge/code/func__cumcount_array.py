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
@final
def _cumcount_array(self, ascending: bool=True) -> np.ndarray:
    """
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Notes
        -----
        this is currently implementing sort=False
        (though the default is sort=True) for groupby in general
        """
    ids, _, ngroups = self._grouper.group_info
    sorter = get_group_index_sorter(ids, ngroups)
    ids, count = (ids[sorter], len(ids))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    run = np.r_[True, ids[:-1] != ids[1:]]
    rep = np.diff(np.r_[np.nonzero(run)[0], count])
    out = (~run).cumsum()
    if ascending:
        out -= np.repeat(out[run], rep)
    else:
        out = np.repeat(out[np.r_[run[1:], True]], rep) - out
    if self._grouper.has_dropped_na:
        out = np.where(ids == -1, np.nan, out.astype(np.float64, copy=False))
    else:
        out = out.astype(np.int64, copy=False)
    rev = np.empty(count, dtype=np.intp)
    rev[sorter] = np.arange(count, dtype=np.intp)
    return out[rev]