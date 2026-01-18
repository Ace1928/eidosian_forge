import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
@staticmethod
def _get_dtype_cmp_class(dtype):
    """
        Get a comparison class name for specified data type.

        Values of different comparison classes cannot be compared.

        Parameters
        ----------
        dtype : dtype
            A data type of a compared value.

        Returns
        -------
        str
            The comparison class name.
        """
    if is_numeric_dtype(dtype) or is_bool_dtype(dtype):
        return 'numeric'
    if is_string_dtype(dtype) or isinstance(dtype, pandas.CategoricalDtype):
        return 'string'
    if is_datetime64_any_dtype(dtype):
        return 'datetime'
    return 'other'