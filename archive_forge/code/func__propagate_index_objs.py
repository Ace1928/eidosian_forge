import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def _propagate_index_objs(self, axis=None):
    """
        Synchronize labels by applying the index object for specific `axis` to the `self._partitions` lazily.

        Adds `set_axis` function to call-queue of each partition from `self._partitions`
        to apply new axis.

        Parameters
        ----------
        axis : int, default: None
            The axis to apply to. If it's None applies to both axes.
        """
    self._filter_empties(compute_metadata=False)
    if axis is None or axis == 0:
        cum_row_lengths = np.cumsum([0] + self.row_lengths)
    if axis is None or axis == 1:
        cum_col_widths = np.cumsum([0] + self.column_widths)
    if axis is None:

        def apply_idx_objs(df, idx, cols):
            return df.set_axis(idx, axis='index').set_axis(cols, axis='columns', copy=False)
        self._partitions = np.array([[self._partitions[i][j].add_to_apply_calls(apply_idx_objs, idx=self.index[slice(cum_row_lengths[i], cum_row_lengths[i + 1])], cols=self.columns[slice(cum_col_widths[j], cum_col_widths[j + 1])], length=self.row_lengths[i], width=self.column_widths[j]) for j in range(len(self._partitions[i]))] for i in range(len(self._partitions))])
        self._deferred_index = False
        self._deferred_column = False
    elif axis == 0:

        def apply_idx_objs(df, idx):
            return df.set_axis(idx, axis='index')
        self._partitions = np.array([[self._partitions[i][j].add_to_apply_calls(apply_idx_objs, idx=self.index[slice(cum_row_lengths[i], cum_row_lengths[i + 1])], length=self.row_lengths[i], width=self.column_widths[j] if self._column_widths_cache is not None else None) for j in range(len(self._partitions[i]))] for i in range(len(self._partitions))])
        self._deferred_index = False
    elif axis == 1:

        def apply_idx_objs(df, cols):
            return df.set_axis(cols, axis='columns')
        self._partitions = np.array([[self._partitions[i][j].add_to_apply_calls(apply_idx_objs, cols=self.columns[slice(cum_col_widths[j], cum_col_widths[j + 1])], length=self.row_lengths[i] if self._row_lengths_cache is not None else None, width=self.column_widths[j]) for j in range(len(self._partitions[i]))] for i in range(len(self._partitions))])
        self._deferred_column = False
    else:
        ErrorMessage.catch_bugs_and_request_email(axis is not None and axis not in [0, 1])