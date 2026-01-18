from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def _reorder_indexer(self, seq: tuple[Scalar | Iterable | AnyArrayLike, ...], indexer: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """
        Reorder an indexer of a MultiIndex (self) so that the labels are in the
        same order as given in seq

        Parameters
        ----------
        seq : label/slice/list/mask or a sequence of such
        indexer: a position indexer of self

        Returns
        -------
        indexer : a sorted position indexer of self ordered as seq
        """
    need_sort = False
    for i, k in enumerate(seq):
        if com.is_null_slice(k) or com.is_bool_indexer(k) or is_scalar(k):
            pass
        elif is_list_like(k):
            if len(k) <= 1:
                pass
            elif self._is_lexsorted():
                k_codes = self.levels[i].get_indexer(k)
                k_codes = k_codes[k_codes >= 0]
                need_sort = (k_codes[:-1] > k_codes[1:]).any()
            else:
                need_sort = True
        elif isinstance(k, slice):
            if self._is_lexsorted():
                need_sort = k.step is not None and k.step < 0
            else:
                need_sort = True
        else:
            need_sort = True
        if need_sort:
            break
    if not need_sort:
        return indexer
    n = len(self)
    keys: tuple[np.ndarray, ...] = ()
    for i, k in enumerate(seq):
        if is_scalar(k):
            k = [k]
        if com.is_bool_indexer(k):
            new_order = np.arange(n)[indexer]
        elif is_list_like(k):
            if not isinstance(k, (np.ndarray, ExtensionArray, Index, ABCSeries)):
                k = sanitize_array(k, None)
            k = algos.unique(k)
            key_order_map = np.ones(len(self.levels[i]), dtype=np.uint64) * len(self.levels[i])
            level_indexer = self.levels[i].get_indexer(k)
            level_indexer = level_indexer[level_indexer >= 0]
            key_order_map[level_indexer] = np.arange(len(level_indexer))
            new_order = key_order_map[self.codes[i][indexer]]
        elif isinstance(k, slice) and k.step is not None and (k.step < 0):
            new_order = np.arange(n)[::-1][indexer]
        elif isinstance(k, slice) and k.start is None and (k.stop is None):
            new_order = np.ones((n,), dtype=np.intp)[indexer]
        else:
            new_order = np.arange(n)[indexer]
        keys = (new_order,) + keys
    ind = np.lexsort(keys)
    return indexer[ind]