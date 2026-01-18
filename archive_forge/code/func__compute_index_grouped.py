from __future__ import annotations
import warnings
from collections.abc import Iterable
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
import pandas.core.common as com
import pandas.core.groupby
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.dtypes.common import (
from pandas.errors import SpecificationError
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import (
from .series import Series
from .utils import is_label
from .window import RollingGroupby
def _compute_index_grouped(self, numerical=False):
    """
        Construct an index of group IDs.

        Parameters
        ----------
        numerical : bool, default: False
            Whether a group indices should be positional (True) or label-based (False).

        Returns
        -------
        dict
            A dict of {group name -> group indices} values.

        See Also
        --------
        pandas.core.groupby.GroupBy.groups
        """
    ErrorMessage.default_to_pandas('Group indices computation')
    by = None
    level = []
    if self._level is not None:
        level = self._level
        if not isinstance(level, list):
            level = [level]
    elif isinstance(self._by, list):
        by = []
        for o in self._by:
            if hashable(o) and o in self._query_compiler.get_index_names(self._axis):
                level.append(o)
            else:
                by.append(o)
    else:
        by = self._by
    is_multi_by = self._is_multi_by or (by is not None and len(level) > 0)
    dropna = self._kwargs.get('dropna', True)
    if isinstance(self._by, BaseQueryCompiler) and is_multi_by:
        by = list(self._by.columns)
    if is_multi_by:
        ErrorMessage.catch_bugs_and_request_email(self._axis == 1)
        if isinstance(by, list) and all((is_label(self._df, o, self._axis) for o in by)):
            pandas_df = self._df._query_compiler.getitem_column_array(by).to_pandas()
        else:
            by = try_cast_to_pandas(by, squeeze=True)
            pandas_df = self._df._to_pandas()
        by = wrap_into_list(by, level)
        groupby_obj = pandas_df.groupby(by=by, dropna=dropna)
        return groupby_obj.indices if numerical else groupby_obj.groups
    else:
        if isinstance(self._by, type(self._query_compiler)):
            by = self._by.to_pandas().squeeze().values
        elif self._by is None:
            index = self._query_compiler.get_axis(self._axis)
            levels_to_drop = [i for i, name in enumerate(index.names) if name not in level and i not in level]
            by = index.droplevel(levels_to_drop)
            if isinstance(by, pandas.MultiIndex):
                by = by.reorder_levels(level)
        else:
            by = self._by
        axis_labels = self._query_compiler.get_axis(self._axis)
        if numerical:
            axis_labels = pandas.RangeIndex(len(axis_labels))
        if dropna:
            return axis_labels.groupby(by)
        else:
            groupby_obj = axis_labels.to_series().groupby(by, dropna=dropna)
            return groupby_obj.indices if numerical else groupby_obj.groups