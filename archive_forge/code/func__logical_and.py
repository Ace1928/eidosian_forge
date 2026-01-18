from inspect import signature
from math import prod
import numpy
import pandas
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_bool_dtype, is_list_like, is_numeric_dtype
import modin.pandas as pd
from modin.core.dataframe.algebra import Binary, Map, Reduce
from modin.error_message import ErrorMessage
from .utils import try_convert_from_interoperable_type
def _logical_and(self, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    return self._logical_binop('_logical_and', x2, out, where, casting, order, dtype, subok)