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
def _get_engine_target(self) -> ArrayLike:
    """
        Get the ndarray or ExtensionArray that we can pass to the IndexEngine
        constructor.
        """
    vals = self._values
    if isinstance(vals, StringArray):
        return vals._ndarray
    if isinstance(vals, ArrowExtensionArray) and self.dtype.kind in 'Mm':
        import pyarrow as pa
        pa_type = vals._pa_array.type
        if pa.types.is_timestamp(pa_type):
            vals = vals._to_datetimearray()
            return vals._ndarray.view('i8')
        elif pa.types.is_duration(pa_type):
            vals = vals._to_timedeltaarray()
            return vals._ndarray.view('i8')
    if type(self) is Index and isinstance(self._values, ExtensionArray) and (not isinstance(self._values, BaseMaskedArray)) and (not (isinstance(self._values, ArrowExtensionArray) and is_numeric_dtype(self.dtype) and (self.dtype.kind != 'O'))):
        return self._values.astype(object)
    return vals