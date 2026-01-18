from __future__ import annotations
from collections.abc import (
import datetime
from functools import (
import inspect
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config.config import option_context
from pandas._libs import (
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core._numba import executor
from pandas.core.apply import warn_alias_replacement
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
from pandas.core.indexes.api import (
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
def _idxmax_idxmin(self, how: Literal['idxmax', 'idxmin'], ignore_unobserved: bool=False, axis: Axis | None | lib.NoDefault=lib.no_default, skipna: bool=True, numeric_only: bool=False) -> NDFrameT:
    """Compute idxmax/idxmin.

        Parameters
        ----------
        how : {'idxmin', 'idxmax'}
            Whether to compute idxmin or idxmax.
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ignore_unobserved : bool, default False
            When True and an unobserved group is encountered, do not raise. This used
            for transform where unobserved groups do not play an impact on the result.

        Returns
        -------
        Series or DataFrame
            idxmax or idxmin for the groupby operation.
        """
    if axis is not lib.no_default:
        if axis is None:
            axis = self.axis
        axis = self.obj._get_axis_number(axis)
        self._deprecate_axis(axis, how)
    else:
        axis = self.axis
    if not self.observed and any((ping._passed_categorical for ping in self._grouper.groupings)):
        expected_len = np.prod([len(ping._group_index) for ping in self._grouper.groupings])
        if len(self._grouper.groupings) == 1:
            result_len = len(self._grouper.groupings[0].grouping_vector.unique())
        else:
            result_len = len(self._grouper.result_index)
        assert result_len <= expected_len
        has_unobserved = result_len < expected_len
        raise_err: bool | np.bool_ = not ignore_unobserved and has_unobserved
        data = self._obj_with_exclusions
        if raise_err and isinstance(data, DataFrame):
            if numeric_only:
                data = data._get_numeric_data()
            raise_err = len(data.columns) > 0
        if raise_err:
            raise ValueError(f"Can't get {how} of an empty group due to unobserved categories. Specify observed=True in groupby instead.")
    elif not skipna:
        if self._obj_with_exclusions.isna().any(axis=None):
            warnings.warn(f'The behavior of {type(self).__name__}.{how} with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError', FutureWarning, stacklevel=find_stack_level())
    if axis == 1:
        try:

            def func(df):
                method = getattr(df, how)
                return method(axis=axis, skipna=skipna, numeric_only=numeric_only)
            func.__name__ = how
            result = self._python_apply_general(func, self._obj_with_exclusions, not_indexed_same=True)
        except ValueError as err:
            name = 'argmax' if how == 'idxmax' else 'argmin'
            if f'attempt to get {name} of an empty sequence' in str(err):
                raise ValueError(f"Can't get {how} of an empty group due to unobserved categories. Specify observed=True in groupby instead.") from None
            raise
        return result
    result = self._agg_general(numeric_only=numeric_only, min_count=1, alias=how, skipna=skipna)
    return result