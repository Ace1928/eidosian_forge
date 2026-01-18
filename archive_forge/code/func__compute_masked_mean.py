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
def _compute_masked_mean(self, mask, output_dtype, axis):
    target = mask.where(self, numpy.nan)._query_compiler
    target = target.astype({col_name: output_dtype for col_name in target.columns}).mean(axis=axis)
    na_propagation_mask = mask.where(self, 0)._query_compiler
    na_propagation_mask = na_propagation_mask.sum(axis=axis, skipna=False)
    target = target.where(na_propagation_mask.notna(), numpy.nan)
    return target