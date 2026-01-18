from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import (
def _intersection(self, other: Index, sort: bool=False):
    """
        intersection specialized to the case with matching dtypes.
        """
    if self.is_monotonic_increasing and other.is_monotonic_increasing and self._can_use_libjoin and other._can_use_libjoin:
        try:
            res_indexer, indexer, _ = self._inner_indexer(other)
        except TypeError:
            pass
        else:
            if is_numeric_dtype(self.dtype):
                res = algos.unique1d(res_indexer)
            else:
                result = self.take(indexer)
                res = result.drop_duplicates()
            return ensure_wrapped_if_datetimelike(res)
    res_values = self._intersection_via_get_indexer(other, sort=sort)
    res_values = _maybe_try_sort(res_values, sort)
    return res_values