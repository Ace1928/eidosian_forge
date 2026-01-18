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
def _determine_setitem_axis(self, row_lookup, col_lookup, row_scalar, col_scalar):
    """
        Determine an axis along which we should do an assignment.

        Parameters
        ----------
        row_lookup : slice or list
            Indexer for rows.
        col_lookup : slice or list
            Indexer for columns.
        row_scalar : bool
            Whether indexer for rows is scalar or not.
        col_scalar : bool
            Whether indexer for columns is scalar or not.

        Returns
        -------
        int or None
            None if this will be a both axis assignment, number of axis to assign in other cases.

        Notes
        -----
        axis = 0: column assignment df[col] = item
        axis = 1: row assignment df.loc[row] = item
        axis = None: assignment along both axes
        """
    if self.arr.shape == (1, 1):
        return None if not row_scalar ^ col_scalar else 1 if row_scalar else 0

    def get_axis(axis):
        return self.arr._query_compiler.index if axis == 0 else self.arr._query_compiler.columns
    row_lookup_len, col_lookup_len = [len(lookup) if not isinstance(lookup, slice) else compute_sliced_len(lookup, len(get_axis(i))) for i, lookup in enumerate([row_lookup, col_lookup])]
    if col_lookup_len == 1 and row_lookup_len == 1:
        axis = None
    elif row_lookup_len == len(self.arr._query_compiler.index) and col_lookup_len == 1 and (self.arr._ndim == 2):
        axis = 0
    elif col_lookup_len == len(self.arr._query_compiler.columns) and row_lookup_len == 1:
        axis = 1
    else:
        axis = None
    return axis