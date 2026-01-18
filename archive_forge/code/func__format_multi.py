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
def _format_multi(self, *, include_names: bool, sparsify: bool | None | lib.NoDefault, formatter: Callable | None=None) -> list:
    if len(self) == 0:
        return []
    stringified_levels = []
    for lev, level_codes in zip(self.levels, self.codes):
        na = _get_na_rep(lev.dtype)
        if len(lev) > 0:
            taken = formatted = lev.take(level_codes)
            formatted = taken._format_flat(include_name=False, formatter=formatter)
            mask = level_codes == -1
            if mask.any():
                formatted = np.array(formatted, dtype=object)
                formatted[mask] = na
                formatted = formatted.tolist()
        else:
            formatted = [pprint_thing(na if isna(x) else x, escape_chars=('\t', '\r', '\n')) for x in algos.take_nd(lev._values, level_codes)]
        stringified_levels.append(formatted)
    result_levels = []
    for lev, lev_name in zip(stringified_levels, self.names):
        level = []
        if include_names:
            level.append(pprint_thing(lev_name, escape_chars=('\t', '\r', '\n')) if lev_name is not None else '')
        level.extend(np.array(lev, dtype=object))
        result_levels.append(level)
    if sparsify is None:
        sparsify = get_option('display.multi_sparse')
    if sparsify:
        sentinel: Literal[''] | bool | lib.NoDefault = ''
        assert isinstance(sparsify, bool) or sparsify is lib.no_default
        if sparsify is lib.no_default:
            sentinel = sparsify
        result_levels = sparsify_labels(result_levels, start=int(include_names), sentinel=sentinel)
    return result_levels