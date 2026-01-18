from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
@cache_readonly
def _get_concat_axis(self) -> Index:
    """
        Return index to be used along concatenation axis.
        """
    if self._is_series:
        if self.bm_axis == 0:
            indexes = [x.index for x in self.objs]
        elif self.ignore_index:
            idx = default_index(len(self.objs))
            return idx
        elif self.keys is None:
            names: list[Hashable] = [None] * len(self.objs)
            num = 0
            has_names = False
            for i, x in enumerate(self.objs):
                if x.ndim != 1:
                    raise TypeError(f"Cannot concatenate type 'Series' with object of type '{type(x).__name__}'")
                if x.name is not None:
                    names[i] = x.name
                    has_names = True
                else:
                    names[i] = num
                    num += 1
            if has_names:
                return Index(names)
            else:
                return default_index(len(self.objs))
        else:
            return ensure_index(self.keys).set_names(self.names)
    else:
        indexes = [x.axes[self.axis] for x in self.objs]
    if self.ignore_index:
        idx = default_index(sum((len(i) for i in indexes)))
        return idx
    if self.keys is None:
        if self.levels is not None:
            raise ValueError('levels supported only when keys is not None')
        concat_axis = _concat_indexes(indexes)
    else:
        concat_axis = _make_concat_multiindex(indexes, self.keys, self.levels, self.names)
    self._maybe_check_integrity(concat_axis)
    return concat_axis