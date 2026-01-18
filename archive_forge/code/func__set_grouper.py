from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
def _set_grouper(self, obj: NDFrameT, sort: bool=False, *, gpr_index: Index | None=None) -> tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
    obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
    if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in 'Mm':
        self._arrow_dtype = ax.dtype
        ax = Index(cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array())
    return (obj, ax, indexer)