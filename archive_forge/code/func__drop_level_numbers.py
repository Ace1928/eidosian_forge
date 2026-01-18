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
def _drop_level_numbers(self, levnums: list[int]):
    """
        Drop MultiIndex levels by level _number_, not name.
        """
    if not levnums and (not isinstance(self, ABCMultiIndex)):
        return self
    if len(levnums) >= self.nlevels:
        raise ValueError(f'Cannot remove {len(levnums)} levels from an index with {self.nlevels} levels: at least one level must be left.')
    self = cast('MultiIndex', self)
    new_levels = list(self.levels)
    new_codes = list(self.codes)
    new_names = list(self.names)
    for i in levnums:
        new_levels.pop(i)
        new_codes.pop(i)
        new_names.pop(i)
    if len(new_levels) == 1:
        lev = new_levels[0]
        if len(lev) == 0:
            if len(new_codes[0]) == 0:
                result = lev[:0]
            else:
                res_values = algos.take(lev._values, new_codes[0], allow_fill=True)
                result = lev._constructor._simple_new(res_values, name=new_names[0])
        else:
            mask = new_codes[0] == -1
            result = new_levels[0].take(new_codes[0])
            if mask.any():
                result = result.putmask(mask, np.nan)
            result._name = new_names[0]
        return result
    else:
        from pandas.core.indexes.multi import MultiIndex
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)