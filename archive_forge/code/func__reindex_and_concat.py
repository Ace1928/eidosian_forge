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
@final
def _reindex_and_concat(self, join_index: Index, left_indexer: npt.NDArray[np.intp] | None, right_indexer: npt.NDArray[np.intp] | None, copy: bool | None) -> DataFrame:
    """
        reindex along index and concat along columns.
        """
    left = self.left[:]
    right = self.right[:]
    llabels, rlabels = _items_overlap_with_suffix(self.left._info_axis, self.right._info_axis, self.suffixes)
    if left_indexer is not None and (not is_range_indexer(left_indexer, len(left))):
        lmgr = left._mgr.reindex_indexer(join_index, left_indexer, axis=1, copy=False, only_slice=True, allow_dups=True, use_na_proxy=True)
        left = left._constructor_from_mgr(lmgr, axes=lmgr.axes)
    left.index = join_index
    if right_indexer is not None and (not is_range_indexer(right_indexer, len(right))):
        rmgr = right._mgr.reindex_indexer(join_index, right_indexer, axis=1, copy=False, only_slice=True, allow_dups=True, use_na_proxy=True)
        right = right._constructor_from_mgr(rmgr, axes=rmgr.axes)
    right.index = join_index
    from pandas import concat
    left.columns = llabels
    right.columns = rlabels
    result = concat([left, right], axis=1, copy=copy)
    return result