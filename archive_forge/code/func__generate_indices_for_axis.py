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
def _generate_indices_for_axis(axis_size, num_elements=numpy.get_printoptions()['edgeitems']):
    if axis_size > num_elements * 2:
        return list(range(num_elements + 1)) + list(range(axis_size - num_elements, axis_size))
    return list(range(axis_size))