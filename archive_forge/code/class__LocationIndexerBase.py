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
class _LocationIndexerBase(ClassLogger):
    """
    Base class for location indexer like loc and iloc.

    Parameters
    ----------
    modin_df : Union[DataFrame, Series]
        DataFrame to operate on.
    """
    df: Union[DataFrame, Series]
    qc: BaseQueryCompiler

    def __init__(self, modin_df: Union[DataFrame, Series]):
        self.df = modin_df
        self.qc = modin_df._query_compiler

    def _validate_key_length(self, key: tuple) -> tuple:
        if len(key) > self.df.ndim:
            if key[0] is Ellipsis:
                key = key[1:]
                if Ellipsis in key:
                    raise IndexingError(_one_ellipsis_message)
                return self._validate_key_length(key)
            raise IndexingError('Too many indexers')
        return key

    def __getitem__(self, key):
        """
        Retrieve dataset according to `key`.

        Parameters
        ----------
        key : callable, scalar, or tuple
            The global row index to retrieve data from.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            Located dataset.

        See Also
        --------
        pandas.DataFrame.loc
        """
        raise NotImplementedError('Implemented by subclasses')

    def __setitem__(self, key, item):
        """
        Assign `item` value to dataset located by `key`.

        Parameters
        ----------
        key : callable or tuple
            The global row numbers to assign data to.
        item : modin.pandas.DataFrame, modin.pandas.Series or scalar
            Value that should be assigned to located dataset.

        See Also
        --------
        pandas.DataFrame.iloc
        """
        raise NotImplementedError('Implemented by subclasses')

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

    def _setitem_positional(self, row_lookup, col_lookup, item, axis=None):
        """
        Assign `item` value to located dataset.

        Parameters
        ----------
        row_lookup : slice or scalar
            The global row index to write item to.
        col_lookup : slice or scalar
            The global col index to write item to.
        item : DataFrame, Series or scalar
            The new item needs to be set. It can be any shape that's
            broadcast-able to the product of the lookup tables.
        axis : {None, 0, 1}, default: None
            If not None, it means that whole axis is used to assign a value.
            0 means assign to whole column, 1 means assign to whole row.
            If None, it means that partial assignment is done on both axes.
        """
        if isinstance(row_lookup, slice):
            row_lookup = range(len(self.qc.index))[row_lookup]
        if isinstance(col_lookup, slice):
            col_lookup = range(len(self.qc.columns))[col_lookup]
        if axis == 0:
            assert len(col_lookup) == 1
            self.df[self.df.columns[col_lookup][0]] = item
        elif axis == 1:
            if hasattr(item, '_query_compiler'):
                if isinstance(item, DataFrame):
                    item = item.squeeze(axis=0)
                item = item._query_compiler
            assert len(row_lookup) == 1
            new_qc = self.qc.setitem(1, self.qc.index[row_lookup[0]], item)
            self.df._create_or_update_from_compiler(new_qc, inplace=True)
            self.qc = self.df._query_compiler
        else:
            new_qc = self.qc.write_items(row_lookup, col_lookup, item)
            self.df._create_or_update_from_compiler(new_qc, inplace=True)
            self.qc = self.df._query_compiler

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
        if self.df.shape == (1, 1):
            return None if not row_scalar ^ col_scalar else 1 if row_scalar else 0

        def get_axis(axis):
            return self.qc.index if axis == 0 else self.qc.columns
        row_lookup_len, col_lookup_len = [len(lookup) if not isinstance(lookup, slice) else compute_sliced_len(lookup, len(get_axis(i))) for i, lookup in enumerate([row_lookup, col_lookup])]
        if col_lookup_len == 1 and row_lookup_len == 1:
            axis = None
        elif row_lookup_len == len(self.qc.index) and col_lookup_len == 1 and isinstance(self.df, DataFrame):
            axis = 0
        elif col_lookup_len == len(self.qc.columns) and row_lookup_len == 1:
            axis = 1
        else:
            axis = None
        return axis

    def _parse_row_and_column_locators(self, tup):
        """
        Unpack the user input for getitem and setitem and compute ndim.

        loc[a] -> ([a], :), 1D
        loc[[a,b]] -> ([a,b], :),
        loc[a,b] -> ([a], [b]), 0D

        Parameters
        ----------
        tup : tuple
            User input to unpack.

        Returns
        -------
        row_loc : scalar or list
            Row locator(s) as a scalar or List.
        col_list : scalar or list
            Column locator(s) as a scalar or List.
        ndim : {0, 1, 2}
            Number of dimensions of located dataset.
        """
        row_loc, col_loc = (slice(None), slice(None))
        if is_tuple(tup):
            row_loc = tup[0]
            if len(tup) == 2:
                col_loc = tup[1]
            if len(tup) > 2:
                raise IndexingError('Too many indexers')
        else:
            row_loc = tup
        row_loc = row_loc(self.df) if callable(row_loc) else row_loc
        col_loc = col_loc(self.df) if callable(col_loc) else col_loc
        return (row_loc, col_loc, _compute_ndim(row_loc, col_loc))

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