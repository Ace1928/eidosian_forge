from __future__ import annotations
from abc import (
from collections.abc import (
from typing import (
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
from pandas.plotting._matplotlib.tools import (
@final
@staticmethod
def _validate_subplots_kwarg(subplots: bool | Sequence[Sequence[str]], data: Series | DataFrame, kind: str) -> bool | list[tuple[int, ...]]:
    """
        Validate the subplots parameter

        - check type and content
        - check for duplicate columns
        - check for invalid column names
        - convert column names into indices
        - add missing columns in a group of their own
        See comments in code below for more details.

        Parameters
        ----------
        subplots : subplots parameters as passed to PlotAccessor

        Returns
        -------
        validated subplots : a bool or a list of tuples of column indices. Columns
        in the same tuple will be grouped together in the resulting plot.
        """
    if isinstance(subplots, bool):
        return subplots
    elif not isinstance(subplots, Iterable):
        raise ValueError('subplots should be a bool or an iterable')
    supported_kinds = ('line', 'bar', 'barh', 'hist', 'kde', 'density', 'area', 'pie')
    if kind not in supported_kinds:
        raise ValueError(f'When subplots is an iterable, kind must be one of {', '.join(supported_kinds)}. Got {kind}.')
    if isinstance(data, ABCSeries):
        raise NotImplementedError('An iterable subplots for a Series is not supported.')
    columns = data.columns
    if isinstance(columns, ABCMultiIndex):
        raise NotImplementedError('An iterable subplots for a DataFrame with a MultiIndex column is not supported.')
    if columns.nunique() != len(columns):
        raise NotImplementedError('An iterable subplots for a DataFrame with non-unique column labels is not supported.')
    out = []
    seen_columns: set[Hashable] = set()
    for group in subplots:
        if not is_list_like(group):
            raise ValueError('When subplots is an iterable, each entry should be a list/tuple of column names.')
        idx_locs = columns.get_indexer_for(group)
        if (idx_locs == -1).any():
            bad_labels = np.extract(idx_locs == -1, group)
            raise ValueError(f'Column label(s) {list(bad_labels)} not found in the DataFrame.')
        unique_columns = set(group)
        duplicates = seen_columns.intersection(unique_columns)
        if duplicates:
            raise ValueError(f'Each column should be in only one subplot. Columns {duplicates} were found in multiple subplots.')
        seen_columns = seen_columns.union(unique_columns)
        out.append(tuple(idx_locs))
    unseen_columns = columns.difference(seen_columns)
    for column in unseen_columns:
        idx_loc = columns.get_loc(column)
        out.append((idx_loc,))
    return out