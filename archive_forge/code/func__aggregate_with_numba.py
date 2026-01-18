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
def _aggregate_with_numba(self, func, *args, engine_kwargs=None, **kwargs):
    """
        Perform groupby aggregation routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
    data = self._obj_with_exclusions
    df = data if data.ndim == 2 else data.to_frame()
    starts, ends, sorted_index, sorted_data = self._numba_prep(df)
    numba_.validate_udf(func)
    numba_agg_func = numba_.generate_numba_agg_func(func, **get_jit_arguments(engine_kwargs, kwargs))
    result = numba_agg_func(sorted_data, sorted_index, starts, ends, len(df.columns), *args)
    index = self._grouper.result_index
    if data.ndim == 1:
        result_kwargs = {'name': data.name}
        result = result.ravel()
    else:
        result_kwargs = {'columns': data.columns}
    res = data._constructor(result, index=index, **result_kwargs)
    if not self.as_index:
        res = self._insert_inaxis_grouper(res)
        res.index = default_index(len(res))
    return res