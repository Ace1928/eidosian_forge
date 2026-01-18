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
def _get_nearest_indexer(self, target: Index, limit: int | None, tolerance) -> npt.NDArray[np.intp]:
    """
        Get the indexer for the nearest index labels; requires an index with
        values that can be subtracted from each other (e.g., not strings or
        tuples).
        """
    if not len(self):
        return self._get_fill_indexer(target, 'pad')
    left_indexer = self.get_indexer(target, 'pad', limit=limit)
    right_indexer = self.get_indexer(target, 'backfill', limit=limit)
    left_distances = self._difference_compat(target, left_indexer)
    right_distances = self._difference_compat(target, right_indexer)
    op = operator.lt if self.is_monotonic_increasing else operator.le
    indexer = np.where(op(left_distances, right_distances) | (right_indexer == -1), left_indexer, right_indexer)
    if tolerance is not None:
        indexer = self._filter_indexer_tolerance(target, indexer, tolerance)
    return indexer