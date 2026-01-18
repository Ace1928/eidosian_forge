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
@final
def _join_monotonic(self, other: Index, how: JoinHow='left') -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    assert other.dtype == self.dtype
    assert self._can_use_libjoin and other._can_use_libjoin
    if self.equals(other):
        ret_index = other if how == 'right' else self
        return (ret_index, None, None)
    ridx: npt.NDArray[np.intp] | None
    lidx: npt.NDArray[np.intp] | None
    if self.is_unique and other.is_unique:
        if how == 'left':
            join_index = self
            lidx = None
            ridx = self._left_indexer_unique(other)
        elif how == 'right':
            join_index = other
            lidx = other._left_indexer_unique(self)
            ridx = None
        elif how == 'inner':
            join_array, lidx, ridx = self._inner_indexer(other)
            join_index = self._wrap_joined_index(join_array, other, lidx, ridx)
        elif how == 'outer':
            join_array, lidx, ridx = self._outer_indexer(other)
            join_index = self._wrap_joined_index(join_array, other, lidx, ridx)
    else:
        if how == 'left':
            join_array, lidx, ridx = self._left_indexer(other)
        elif how == 'right':
            join_array, ridx, lidx = other._left_indexer(self)
        elif how == 'inner':
            join_array, lidx, ridx = self._inner_indexer(other)
        elif how == 'outer':
            join_array, lidx, ridx = self._outer_indexer(other)
        assert lidx is not None
        assert ridx is not None
        join_index = self._wrap_joined_index(join_array, other, lidx, ridx)
    lidx = None if lidx is None else ensure_platform_int(lidx)
    ridx = None if ridx is None else ensure_platform_int(ridx)
    return (join_index, lidx, ridx)