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
def _maybe_downcast_for_indexing(self, other: Index) -> tuple[Index, Index]:
    """
        When dealing with an object-dtype Index and a non-object Index, see
        if we can upcast the object-dtype one to improve performance.
        """
    if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex):
        if self.tz is not None and other.tz is not None and (not tz_compare(self.tz, other.tz)):
            return (self.tz_convert('UTC'), other.tz_convert('UTC'))
    elif self.inferred_type == 'date' and isinstance(other, ABCDatetimeIndex):
        try:
            return (type(other)(self), other)
        except OutOfBoundsDatetime:
            return (self, other)
    elif self.inferred_type == 'timedelta' and isinstance(other, ABCTimedeltaIndex):
        return (type(other)(self), other)
    elif self.dtype.kind == 'u' and other.dtype.kind == 'i':
        if other.min() >= 0:
            return (self, other.astype(self.dtype))
    elif self._is_multi and (not other._is_multi):
        try:
            other = type(self).from_tuples(other)
        except (TypeError, ValueError):
            self = Index(self._values)
    if not is_object_dtype(self.dtype) and is_object_dtype(other.dtype):
        other, self = other._maybe_downcast_for_indexing(self)
    return (self, other)