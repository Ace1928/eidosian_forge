from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None:
    from pandas import Series
    if self.on is not None and (not self._on.equals(obj.index)):
        name = self._on.name
        extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
        if name in result.columns:
            result[name] = extra_col
        elif name in result.index.names:
            pass
        elif name in self._selected_obj.columns:
            old_cols = self._selected_obj.columns
            new_cols = result.columns
            old_loc = old_cols.get_loc(name)
            overlap = new_cols.intersection(old_cols[:old_loc])
            new_loc = len(overlap)
            result.insert(new_loc, name, extra_col)
        else:
            result[name] = extra_col