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
def _transform_general(self, func, engine, engine_kwargs, *args, **kwargs):
    if maybe_use_numba(engine):
        return self._transform_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
    from pandas.core.reshape.concat import concat
    applied = []
    obj = self._obj_with_exclusions
    gen = self._grouper.get_iterator(obj, axis=self.axis)
    fast_path, slow_path = self._define_paths(func, *args, **kwargs)
    try:
        name, group = next(gen)
    except StopIteration:
        pass
    else:
        object.__setattr__(group, 'name', name)
        try:
            path, res = self._choose_path(fast_path, slow_path, group)
        except ValueError as err:
            msg = 'transform must return a scalar value for each group'
            raise ValueError(msg) from err
        if group.size > 0:
            res = _wrap_transform_general_frame(self.obj, group, res)
            applied.append(res)
    for name, group in gen:
        if group.size == 0:
            continue
        object.__setattr__(group, 'name', name)
        res = path(group)
        res = _wrap_transform_general_frame(self.obj, group, res)
        applied.append(res)
    concat_index = obj.columns if self.axis == 0 else obj.index
    other_axis = 1 if self.axis == 0 else 0
    concatenated = concat(applied, axis=self.axis, verify_integrity=False)
    concatenated = concatenated.reindex(concat_index, axis=other_axis, copy=False)
    return self._set_result_index_ordered(concatenated)