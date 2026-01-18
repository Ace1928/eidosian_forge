import itertools
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.pandas.indexing import compute_sliced_len, is_range_like, is_slice, is_tuple
from modin.pandas.utils import is_scalar
from .arr import array
def _compute_ndim(row_loc, col_loc):
    """
    Compute the number of dimensions of result from locators.

    Parameters
    ----------
    row_loc : list or scalar
        Row locator.
    col_loc : list or scalar
        Column locator.

    Returns
    -------
    {0, 1, 2}
        Number of dimensions in located dataset.
    """
    row_scalar = is_scalar(row_loc) or is_tuple(row_loc)
    col_scalar = is_scalar(col_loc) or is_tuple(col_loc)
    if row_scalar and col_scalar:
        ndim = 0
    elif row_scalar ^ col_scalar:
        ndim = 1
    else:
        ndim = 2
    return ndim