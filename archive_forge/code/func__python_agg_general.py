from __future__ import annotations
from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import algorithms
from pandas.core.apply import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
from pandas.core.groupby.groupby import (
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby
def _python_agg_general(self, func, *args, **kwargs):
    orig_func = func
    func = com.is_builtin_func(func)
    if orig_func != func:
        alias = com._builtin_table_alias[func]
        warn_alias_replacement(self, orig_func, alias)
    f = lambda x: func(x, *args, **kwargs)
    if self.ngroups == 0:
        return self._python_apply_general(f, self._selected_obj, is_agg=True)
    obj = self._obj_with_exclusions
    if self.axis == 1:
        obj = obj.T
    if not len(obj.columns):
        return self._python_apply_general(f, self._selected_obj)
    output: dict[int, ArrayLike] = {}
    for idx, (name, ser) in enumerate(obj.items()):
        result = self._grouper.agg_series(ser, f)
        output[idx] = result
    res = self.obj._constructor(output)
    res.columns = obj.columns.copy(deep=False)
    return self._wrap_aggregated_output(res)