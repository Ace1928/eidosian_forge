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
def _check_missing_loc(self, row_loc, col_loc):
    """
        Help `__setitem__` compute whether an axis needs appending.

        Parameters
        ----------
        row_loc : scalar, slice, list, array or tuple
            Row locator.
        col_loc : scalar, slice, list, array or tuple
            Columns locator.

        Returns
        -------
        int or None :
            0 if new row, 1 if new column, None if neither.
        """
    if is_scalar(row_loc):
        return 0 if row_loc not in self.qc.index else None
    elif isinstance(row_loc, list):
        missing_labels = self._compute_enlarge_labels(pandas.Index(row_loc), self.qc.index)
        if len(missing_labels) > 1:
            raise KeyError('{} not in index'.format(list(missing_labels)))
    if not (is_list_like(row_loc) or isinstance(row_loc, slice)) and row_loc not in self.qc.index:
        return 0
    if isinstance(col_loc, list) and len(pandas.Index(col_loc).difference(self.qc.columns)) >= 1:
        return 1
    if is_scalar(col_loc) and col_loc not in self.qc.columns:
        return 1
    return None