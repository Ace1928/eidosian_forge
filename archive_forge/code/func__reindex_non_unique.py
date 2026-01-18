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
def _reindex_non_unique(self, target: Index) -> tuple[Index, npt.NDArray[np.intp], npt.NDArray[np.intp] | None]:
    """
        Create a new index with target's values (move/add/delete values as
        necessary) use with non-unique Index and a possibly non-unique target.

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.
        new_indexer : np.ndarray[np.intp] or None

        """
    target = ensure_index(target)
    if len(target) == 0:
        return (self[:0], np.array([], dtype=np.intp), None)
    indexer, missing = self.get_indexer_non_unique(target)
    check = indexer != -1
    new_labels: Index | np.ndarray = self.take(indexer[check])
    new_indexer = None
    if len(missing):
        length = np.arange(len(indexer), dtype=np.intp)
        missing = ensure_platform_int(missing)
        missing_labels = target.take(missing)
        missing_indexer = length[~check]
        cur_labels = self.take(indexer[check]).values
        cur_indexer = length[check]
        new_labels = np.empty((len(indexer),), dtype=object)
        new_labels[cur_indexer] = cur_labels
        new_labels[missing_indexer] = missing_labels
        if not len(self):
            new_indexer = np.arange(0, dtype=np.intp)
        elif target.is_unique:
            new_indexer = np.arange(len(indexer), dtype=np.intp)
            new_indexer[cur_indexer] = np.arange(len(cur_labels))
            new_indexer[missing_indexer] = -1
        else:
            indexer[~check] = -1
            new_indexer = np.arange(len(self.take(indexer)), dtype=np.intp)
            new_indexer[~check] = -1
    if not isinstance(self, ABCMultiIndex):
        new_index = Index(new_labels, name=self.name)
    else:
        new_index = type(self).from_tuples(new_labels, names=self.names)
    return (new_index, indexer, new_indexer)