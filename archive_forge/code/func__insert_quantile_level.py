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
def _insert_quantile_level(idx: Index, qs: npt.NDArray[np.float64]) -> MultiIndex:
    """
    Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

    The quantile level in the MultiIndex is a repeated copy of 'qs'.

    Parameters
    ----------
    idx : Index
    qs : np.ndarray[float64]

    Returns
    -------
    MultiIndex
    """
    nqs = len(qs)
    lev_codes, lev = Index(qs).factorize()
    lev_codes = coerce_indexer_dtype(lev_codes, lev)
    if idx._is_multi:
        idx = cast(MultiIndex, idx)
        levels = list(idx.levels) + [lev]
        codes = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(lev_codes, len(idx))]
        mi = MultiIndex(levels=levels, codes=codes, names=idx.names + [None])
    else:
        nidx = len(idx)
        idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
        levels = [idx, lev]
        codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
        mi = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])
    return mi