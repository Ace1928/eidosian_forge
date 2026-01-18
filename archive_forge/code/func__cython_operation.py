from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def _cython_operation(self, kind: str, values, how: str, axis: AxisInt, min_count: int=-1, **kwargs) -> ArrayLike:
    """
        Returns the values of a cython operation.
        """
    assert kind in ['transform', 'aggregate']
    cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)
    ids, _, _ = self.group_info
    ngroups = self.ngroups
    return cy_op.cython_operation(values=values, axis=axis, min_count=min_count, comp_ids=ids, ngroups=ngroups, **kwargs)