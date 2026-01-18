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
def _set_result_index_ordered(self, result: OutputFrameOrSeries) -> OutputFrameOrSeries:
    obj_axis = self.obj._get_axis(self.axis)
    if self._grouper.is_monotonic and (not self._grouper.has_dropped_na):
        result = result.set_axis(obj_axis, axis=self.axis, copy=False)
        return result
    original_positions = Index(self._grouper.result_ilocs())
    result = result.set_axis(original_positions, axis=self.axis, copy=False)
    result = result.sort_index(axis=self.axis)
    if self._grouper.has_dropped_na:
        result = result.reindex(RangeIndex(len(obj_axis)), axis=self.axis)
    result = result.set_axis(obj_axis, axis=self.axis, copy=False)
    return result