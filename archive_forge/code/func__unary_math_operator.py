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
def _unary_math_operator(self, opName, *args, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    out_dtype = dtype if dtype is not None else out.dtype if out is not None else self.dtype
    check_kwargs(order=order, casting=casting, subok=subok, where=where)
    result = self._query_compiler.astype({col_name: out_dtype for col_name in self._query_compiler.columns})
    result = getattr(result, opName)(*args)
    if dtype is not None:
        result = result.astype({col_name: dtype for col_name in result.columns})
    if out is not None:
        out = try_convert_from_interoperable_type(out)
        check_can_broadcast_to_output(self, out)
        out._query_compiler = result
        return out
    return array(_query_compiler=result, _ndim=self._ndim)