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
def _handle_boolean_masking(self, row_loc, col_loc):
    """
        Retrieve dataset according to the boolean mask for rows and an indexer for columns.

        In comparison with the regular ``loc/iloc.__getitem__`` flow this method efficiently
        masks rows with a Modin Series boolean mask without materializing it (if the selected
        execution implements such masking).

        Parameters
        ----------
        row_loc : modin.pandas.Series of bool dtype
            Boolean mask to index rows with.
        col_loc : object
            An indexer along column axis.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            Located dataset.
        """
    ErrorMessage.catch_bugs_and_request_email(failure_condition=not isinstance(row_loc, Series), extra_log=f'Only ``modin.pandas.Series`` boolean masks are acceptable, got: {type(row_loc)}')
    masked_df = self.df.__constructor__(query_compiler=self.qc.getitem_array(row_loc._query_compiler))
    if isinstance(masked_df, Series):
        assert col_loc == slice(None)
        return masked_df
    return type(self)(masked_df)[slice(None), col_loc]