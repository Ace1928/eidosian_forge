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
def _op_via_apply(self, name: str, *args, **kwargs):
    """Compute the result of an operation by using GroupBy's apply."""
    f = getattr(type(self._obj_with_exclusions), name)
    sig = inspect.signature(f)
    if 'axis' in kwargs and kwargs['axis'] is not lib.no_default:
        axis = self.obj._get_axis_number(kwargs['axis'])
        self._deprecate_axis(axis, name)
    elif 'axis' in kwargs:
        if name == 'skew':
            pass
        elif name == 'fillna':
            kwargs['axis'] = None
        else:
            kwargs['axis'] = 0
    if 'axis' in sig.parameters:
        if kwargs.get('axis', None) is None or kwargs.get('axis') is lib.no_default:
            kwargs['axis'] = self.axis

    def curried(x):
        return f(x, *args, **kwargs)
    curried.__name__ = name
    if name in base.plotting_methods:
        return self._python_apply_general(curried, self._selected_obj)
    is_transform = name in base.transformation_kernels
    result = self._python_apply_general(curried, self._obj_with_exclusions, is_transform=is_transform, not_indexed_same=not is_transform)
    if self._grouper.has_dropped_na and is_transform:
        result = self._set_result_index_ordered(result)
    return result