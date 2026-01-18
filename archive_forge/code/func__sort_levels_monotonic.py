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
def _sort_levels_monotonic(self, raise_if_incomparable: bool=False) -> MultiIndex:
    """
        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )

        >>> mi.sort_values()
        MultiIndex([('a', 'aa'),
                    ('a', 'bb'),
                    ('b', 'aa'),
                    ('b', 'bb')],
                   )
        """
    if self._is_lexsorted() and self.is_monotonic_increasing:
        return self
    new_levels = []
    new_codes = []
    for lev, level_codes in zip(self.levels, self.codes):
        if not lev.is_monotonic_increasing:
            try:
                indexer = lev.argsort()
            except TypeError:
                if raise_if_incomparable:
                    raise
            else:
                lev = lev.take(indexer)
                indexer = ensure_platform_int(indexer)
                ri = lib.get_reverse_indexer(indexer, len(indexer))
                level_codes = algos.take_nd(ri, level_codes, fill_value=-1)
        new_levels.append(lev)
        new_codes.append(level_codes)
    return MultiIndex(new_levels, new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)