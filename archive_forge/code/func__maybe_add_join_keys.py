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
def _maybe_add_join_keys(self, result: DataFrame, left_indexer: npt.NDArray[np.intp] | None, right_indexer: npt.NDArray[np.intp] | None) -> None:
    left_has_missing = None
    right_has_missing = None
    assert all((isinstance(x, _known) for x in self.left_join_keys))
    keys = zip(self.join_names, self.left_on, self.right_on)
    for i, (name, lname, rname) in enumerate(keys):
        if not _should_fill(lname, rname):
            continue
        take_left, take_right = (None, None)
        if name in result:
            if left_indexer is not None or right_indexer is not None:
                if name in self.left:
                    if left_has_missing is None:
                        left_has_missing = False if left_indexer is None else (left_indexer == -1).any()
                    if left_has_missing:
                        take_right = self.right_join_keys[i]
                        if result[name].dtype != self.left[name].dtype:
                            take_left = self.left[name]._values
                elif name in self.right:
                    if right_has_missing is None:
                        right_has_missing = False if right_indexer is None else (right_indexer == -1).any()
                    if right_has_missing:
                        take_left = self.left_join_keys[i]
                        if result[name].dtype != self.right[name].dtype:
                            take_right = self.right[name]._values
        else:
            take_left = self.left_join_keys[i]
            take_right = self.right_join_keys[i]
        if take_left is not None or take_right is not None:
            if take_left is None:
                lvals = result[name]._values
            elif left_indexer is None:
                lvals = take_left
            else:
                take_left = extract_array(take_left, extract_numpy=True)
                lfill = na_value_for_dtype(take_left.dtype)
                lvals = algos.take_nd(take_left, left_indexer, fill_value=lfill)
            if take_right is None:
                rvals = result[name]._values
            elif right_indexer is None:
                rvals = take_right
            else:
                taker = extract_array(take_right, extract_numpy=True)
                rfill = na_value_for_dtype(taker.dtype)
                rvals = algos.take_nd(taker, right_indexer, fill_value=rfill)
            if left_indexer is not None and (left_indexer == -1).all():
                key_col = Index(rvals)
                result_dtype = rvals.dtype
            elif right_indexer is not None and (right_indexer == -1).all():
                key_col = Index(lvals)
                result_dtype = lvals.dtype
            else:
                key_col = Index(lvals)
                if left_indexer is not None:
                    mask_left = left_indexer == -1
                    key_col = key_col.where(~mask_left, rvals)
                result_dtype = find_common_type([lvals.dtype, rvals.dtype])
                if lvals.dtype.kind == 'M' and rvals.dtype.kind == 'M' and (result_dtype.kind == 'O'):
                    result_dtype = key_col.dtype
            if result._is_label_reference(name):
                result[name] = result._constructor_sliced(key_col, dtype=result_dtype, index=result.index)
            elif result._is_level_reference(name):
                if isinstance(result.index, MultiIndex):
                    key_col.name = name
                    idx_list = [result.index.get_level_values(level_name) if level_name != name else key_col for level_name in result.index.names]
                    result.set_index(idx_list, inplace=True)
                else:
                    result.index = Index(key_col, name=name)
            else:
                result.insert(i, name or f'key_{i}', key_col)