from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar
def _helper_for__getitem__(self, key, row_loc, col_loc, ndim):
    """
        Retrieve dataset according to `key`, row_loc, and col_loc.

        Parameters
        ----------
        key : callable, scalar, or tuple
            The global row index to retrieve data from.
        row_loc : callable, scalar, or slice
            Row locator(s) as a scalar or List.
        col_loc : callable, scalar, or slice
            Row locator(s) as a scalar or List.
        ndim : int
            The number of dimensions of the returned object.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            Located dataset.
        """
    row_scalar = is_scalar(row_loc)
    col_scalar = is_scalar(col_loc)
    row_multiindex_full_lookup = self._multiindex_possibly_contains_key(axis=0, key=row_loc)
    col_multiindex_full_lookup = self._multiindex_possibly_contains_key(axis=1, key=col_loc)
    levels_already_dropped = row_multiindex_full_lookup or col_multiindex_full_lookup
    if isinstance(row_loc, Series) and is_boolean_array(row_loc):
        return self._handle_boolean_masking(row_loc, col_loc)
    qc_view = self.qc.take_2d_labels(row_loc, col_loc)
    result = self._get_pandas_object_from_qc_view(qc_view, row_multiindex_full_lookup, col_multiindex_full_lookup, row_scalar, col_scalar, ndim)
    if isinstance(result, Series):
        result._parent = self.df
        result._parent_axis = 0
    col_loc_as_list = [col_loc] if col_scalar else col_loc
    row_loc_as_list = [row_loc] if row_scalar else row_loc
    if isinstance(result, (Series, DataFrame)) and result._query_compiler.has_multiindex() and (not levels_already_dropped):
        if isinstance(result, Series) and (not isinstance(col_loc_as_list, slice)) and all((col_loc_as_list[i] in result.index.levels[i] for i in range(len(col_loc_as_list)))):
            result.index = result.index.droplevel(list(range(len(col_loc_as_list))))
        elif not isinstance(row_loc_as_list, slice) and all((not isinstance(row_loc_as_list[i], slice) and row_loc_as_list[i] in result.index.levels[i] for i in range(len(row_loc_as_list)))):
            result.index = result.index.droplevel(list(range(len(row_loc_as_list))))
    if isinstance(result, DataFrame) and (not isinstance(col_loc_as_list, slice)) and (not levels_already_dropped) and result._query_compiler.has_multiindex(axis=1) and all((col_loc_as_list[i] in result.columns.levels[i] for i in range(len(col_loc_as_list)))):
        result.columns = result.columns.droplevel(list(range(len(col_loc_as_list))))
    if row_loc is not None and isinstance(col_loc, slice) and (col_loc == slice(None)) and isinstance(key, pandas.Index):
        result.index = key
    return result