from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
@doc(GroupByIndexingMixin._positional_selector)
class GroupByPositionalSelector:

    def __init__(self, groupby_object: groupby.GroupBy) -> None:
        self.groupby_object = groupby_object

    def __getitem__(self, arg: PositionalIndexer | tuple) -> DataFrame | Series:
        """
        Select by positional index per group.

        Implements GroupBy._positional_selector

        Parameters
        ----------
        arg : PositionalIndexer | tuple
            Allowed values are:
            - int
            - int valued iterable such as list or range
            - slice with step either None or positive
            - tuple of integers and slices

        Returns
        -------
        Series
            The filtered subset of the original groupby Series.
        DataFrame
            The filtered subset of the original groupby DataFrame.

        See Also
        --------
        DataFrame.iloc : Integer-location based indexing for selection by position.
        GroupBy.head : Return first n rows of each group.
        GroupBy.tail : Return last n rows of each group.
        GroupBy._positional_selector : Return positional selection for each group.
        GroupBy.nth : Take the nth row from each group if n is an int, or a
            subset of rows, if n is a list of ints.
        """
        mask = self.groupby_object._make_mask_from_positional_indexer(arg)
        return self.groupby_object._mask_selected_obj(mask)