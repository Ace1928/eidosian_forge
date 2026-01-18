from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
def _reverse_indexer(self) -> dict[Hashable, npt.NDArray[np.intp]]:
    """
        Compute the inverse of a categorical, returning
        a dict of categories -> indexers.

        *This is an internal function*

        Returns
        -------
        Dict[Hashable, np.ndarray[np.intp]]
            dict of categories -> indexers

        Examples
        --------
        >>> c = pd.Categorical(list('aabca'))
        >>> c
        ['a', 'a', 'b', 'c', 'a']
        Categories (3, object): ['a', 'b', 'c']
        >>> c.categories
        Index(['a', 'b', 'c'], dtype='object')
        >>> c.codes
        array([0, 0, 1, 2, 0], dtype=int8)
        >>> c._reverse_indexer()
        {'a': array([0, 1, 4]), 'b': array([2]), 'c': array([3])}

        """
    categories = self.categories
    r, counts = libalgos.groupsort_indexer(ensure_platform_int(self.codes), categories.size)
    counts = ensure_int64(counts).cumsum()
    _result = (r[start:end] for start, end in zip(counts, counts[1:]))
    return dict(zip(categories, _result))