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
def _multiindex_possibly_contains_key(self, axis, key):
    """
        Determine if a MultiIndex row/column possibly contains a key.

        Check to see if the current DataFrame has a MultiIndex row/column and if it does,
        check to see if the key is potentially a full key-lookup such that the number of
        levels match up with the length of the tuple key.

        Parameters
        ----------
        axis : {0, 1}
            0 for row, 1 for column.
        key : Any
            Lookup key for MultiIndex row/column.

        Returns
        -------
        bool
            If the MultiIndex possibly contains the given key.

        Notes
        -----
        This function only returns False if we have a partial key lookup. It's
        possible that this function returns True for a key that does NOT exist
        since we only check the length of the `key` tuple to match the number
        of levels in the MultiIndex row/colunmn.
        """
    if not self.qc.has_multiindex(axis=axis):
        return False
    multiindex = self.df.index if axis == 0 else self.df.columns
    return isinstance(key, tuple) and len(key) == len(multiindex.levels)