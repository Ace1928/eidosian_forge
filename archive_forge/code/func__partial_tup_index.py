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
def _partial_tup_index(self, tup: tuple, side: Literal['left', 'right']='left'):
    if len(tup) > self._lexsort_depth:
        raise UnsortedIndexError(f'Key length ({len(tup)}) was greater than MultiIndex lexsort depth ({self._lexsort_depth})')
    n = len(tup)
    start, end = (0, len(self))
    zipped = zip(tup, self.levels, self.codes)
    for k, (lab, lev, level_codes) in enumerate(zipped):
        section = level_codes[start:end]
        loc: npt.NDArray[np.intp] | np.intp | int
        if lab not in lev and (not isna(lab)):
            try:
                loc = algos.searchsorted(lev, lab, side=side)
            except TypeError as err:
                raise TypeError(f'Level type mismatch: {lab}') from err
            if not is_integer(loc):
                raise TypeError(f'Level type mismatch: {lab}')
            if side == 'right' and loc >= 0:
                loc -= 1
            return start + algos.searchsorted(section, loc, side=side)
        idx = self._get_loc_single_level_index(lev, lab)
        if isinstance(idx, slice) and k < n - 1:
            start = idx.start
            end = idx.stop
        elif k < n - 1:
            end = start + algos.searchsorted(section, idx, side='right')
            start = start + algos.searchsorted(section, idx, side='left')
        elif isinstance(idx, slice):
            idx = idx.start
            return start + algos.searchsorted(section, idx, side=side)
        else:
            return start + algos.searchsorted(section, idx, side=side)