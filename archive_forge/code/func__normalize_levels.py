from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@staticmethod
def _normalize_levels(columns, reference=None):
    """
        Normalize levels of MultiIndex column names.

        The function fills missing levels with empty strings as pandas do:
        '''
        >>> columns = ["a", ("l1", "l2"), ("l1a", "l2a", "l3a")]
        >>> _normalize_levels(columns)
        [("a", "", ""), ("l1", "l2", ""), ("l1a", "l2a", "l3a")]
        >>> # with a reference
        >>> idx = pandas.MultiIndex(...)
        >>> idx.nlevels
        4
        >>> _normalize_levels(columns, reference=idx)
        [("a", "", "", ""), ("l1", "l2", "", ""), ("l1a", "l2a", "l3a", "")]
        '''

        Parameters
        ----------
        columns : sequence
            Labels to normalize. If dictionary, will replace keys with normalized columns.
        reference : pandas.Index, optional
            An index to match the number of levels with. If reference is a MultiIndex, then the reference number
            of levels should not be greater than the maximum number of levels in `columns`. If not specified,
            the `columns` themselves become a `reference`.

        Returns
        -------
        sequence
            Column values with normalized levels.
        dict[hashable, hashable]
            Mapping from old column names to new names, only contains column names that
            were changed.

        Raises
        ------
        ValueError
            When the reference number of levels is greater than the maximum number of levels
            in `columns`.
        """
    if reference is None:
        reference = columns
    if isinstance(reference, pandas.Index):
        max_nlevels = reference.nlevels
    else:
        max_nlevels = 1
        for col in reference:
            if isinstance(col, tuple):
                max_nlevels = max(max_nlevels, len(col))
    if max_nlevels == 1:
        return (columns, {})
    max_columns_nlevels = 1
    for col in columns:
        if isinstance(col, tuple):
            max_columns_nlevels = max(max_columns_nlevels, len(col))
    if max_columns_nlevels > max_nlevels:
        raise ValueError(f'The reference number of levels is greater than the maximum number of levels in columns: {max_columns_nlevels} > {max_nlevels}')
    new_columns = []
    old_to_new_mapping = {}
    for col in columns:
        old_col = col
        if not isinstance(col, tuple):
            col = (col,)
        col = col + ('',) * (max_nlevels - len(col))
        new_columns.append(col)
        if old_col != col:
            old_to_new_mapping[old_col] = col
    return (new_columns, old_to_new_mapping)