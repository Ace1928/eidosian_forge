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
def _join_empty(self, other: Index, how: JoinHow, sort: bool) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    assert len(self) == 0 or len(other) == 0
    _validate_join_method(how)
    lidx: np.ndarray | None
    ridx: np.ndarray | None
    if len(other):
        how = cast(JoinHow, {'left': 'right', 'right': 'left'}.get(how, how))
        join_index, ridx, lidx = other._join_empty(self, how, sort)
    elif how in ['left', 'outer']:
        if sort and (not self.is_monotonic_increasing):
            lidx = self.argsort()
            join_index = self.take(lidx)
        else:
            lidx = None
            join_index = self._view()
        ridx = np.broadcast_to(np.intp(-1), len(join_index))
    else:
        join_index = other._view()
        lidx = np.array([], dtype=np.intp)
        ridx = None
    return (join_index, lidx, ridx)