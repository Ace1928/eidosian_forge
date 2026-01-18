from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
def _cross_merge(left: DataFrame, right: DataFrame, on: IndexLabel | AnyArrayLike | None=None, left_on: IndexLabel | AnyArrayLike | None=None, right_on: IndexLabel | AnyArrayLike | None=None, left_index: bool=False, right_index: bool=False, sort: bool=False, suffixes: Suffixes=('_x', '_y'), copy: bool | None=None, indicator: str | bool=False, validate: str | None=None) -> DataFrame:
    """
    See merge.__doc__ with how='cross'
    """
    if left_index or right_index or right_on is not None or (left_on is not None) or (on is not None):
        raise MergeError('Can not pass on, right_on, left_on or set right_index=True or left_index=True')
    cross_col = f'_cross_{uuid.uuid4()}'
    left = left.assign(**{cross_col: 1})
    right = right.assign(**{cross_col: 1})
    left_on = right_on = [cross_col]
    res = merge(left, right, how='inner', on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, indicator=indicator, validate=validate, copy=copy)
    del res[cross_col]
    return res