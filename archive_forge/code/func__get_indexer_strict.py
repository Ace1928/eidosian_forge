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
def _get_indexer_strict(self, key, axis_name: str) -> tuple[Index, npt.NDArray[np.intp]]:
    keyarr = key
    if not isinstance(keyarr, Index):
        keyarr = com.asarray_tuplesafe(keyarr)
    if len(keyarr) and (not isinstance(keyarr[0], tuple)):
        indexer = self._get_indexer_level_0(keyarr)
        self._raise_if_missing(key, indexer, axis_name)
        return (self[indexer], indexer)
    return super()._get_indexer_strict(key, axis_name)