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
class _OrderedMerge(_MergeOperation):
    _merge_type = 'ordered_merge'

    def __init__(self, left: DataFrame | Series, right: DataFrame | Series, on: IndexLabel | None=None, left_on: IndexLabel | None=None, right_on: IndexLabel | None=None, left_index: bool=False, right_index: bool=False, suffixes: Suffixes=('_x', '_y'), fill_method: str | None=None, how: JoinHow | Literal['asof']='outer') -> None:
        self.fill_method = fill_method
        _MergeOperation.__init__(self, left, right, on=on, left_on=left_on, left_index=left_index, right_index=right_index, right_on=right_on, how=how, suffixes=suffixes, sort=True)

    def get_result(self, copy: bool | None=True) -> DataFrame:
        join_index, left_indexer, right_indexer = self._get_join_info()
        left_join_indexer: npt.NDArray[np.intp] | None
        right_join_indexer: npt.NDArray[np.intp] | None
        if self.fill_method == 'ffill':
            if left_indexer is None:
                left_join_indexer = None
            else:
                left_join_indexer = libjoin.ffill_indexer(left_indexer)
            if right_indexer is None:
                right_join_indexer = None
            else:
                right_join_indexer = libjoin.ffill_indexer(right_indexer)
        elif self.fill_method is None:
            left_join_indexer = left_indexer
            right_join_indexer = right_indexer
        else:
            raise ValueError("fill_method must be 'ffill' or None")
        result = self._reindex_and_concat(join_index, left_join_indexer, right_join_indexer, copy=copy)
        self._maybe_add_join_keys(result, left_indexer, right_indexer)
        return result