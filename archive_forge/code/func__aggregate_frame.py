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
def _aggregate_frame(self, func, *args, **kwargs) -> DataFrame:
    if self._grouper.nkeys != 1:
        raise AssertionError('Number of keys must be 1')
    obj = self._obj_with_exclusions
    result: dict[Hashable, NDFrame | np.ndarray] = {}
    for name, grp_df in self._grouper.get_iterator(obj, self.axis):
        fres = func(grp_df, *args, **kwargs)
        result[name] = fres
    result_index = self._grouper.result_index
    other_ax = obj.axes[1 - self.axis]
    out = self.obj._constructor(result, index=other_ax, columns=result_index)
    if self.axis == 0:
        out = out.T
    return out