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
def _get_pandas_object_from_qc_view(self, qc_view, row_multiindex_full_lookup: bool, col_multiindex_full_lookup: bool, row_scalar: bool, col_scalar: bool, ndim: int):
    """
        Convert the query compiler view to the appropriate pandas object.

        Parameters
        ----------
        qc_view : BaseQueryCompiler
            Query compiler to convert.
        row_multiindex_full_lookup : bool
            See _multiindex_possibly_contains_key.__doc__.
        col_multiindex_full_lookup : bool
            See _multiindex_possibly_contains_key.__doc__.
        row_scalar : bool
            Whether indexer for rows is scalar.
        col_scalar : bool
            Whether indexer for columns is scalar.
        ndim : {0, 1, 2}
            Number of dimensions in dataset to be retrieved.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            The pandas object with the data from the query compiler view.

        Notes
        -----
        Usage of `slice(None)` as a lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
    if ndim == 2:
        return self.df.__constructor__(query_compiler=qc_view)
    if isinstance(self.df, Series) and (not row_scalar):
        return self.df.__constructor__(query_compiler=qc_view)
    if isinstance(self.df, Series):
        axis = 0
    elif ndim == 0:
        axis = None
    else:
        axis = None if col_scalar and row_scalar or (row_multiindex_full_lookup and col_multiindex_full_lookup) else 1 if col_scalar or col_multiindex_full_lookup else 0
    res_df = self.df.__constructor__(query_compiler=qc_view)
    return res_df.squeeze(axis=axis)