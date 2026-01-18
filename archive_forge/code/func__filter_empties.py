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
def _filter_empties(self, compute_metadata=True):
    """
        Remove empty partitions from `self._partitions` to avoid triggering excess computation.

        Parameters
        ----------
        compute_metadata : bool, default: True
            Trigger the computations for partition sizes and labels if they're not done already.
        """
    if not compute_metadata and (self._row_lengths_cache is None or self._column_widths_cache is None):
        return
    if self.has_materialized_index and len(self.index) == 0 or (self.has_materialized_columns and len(self.columns) == 0) or sum(self.row_lengths) == 0 or (sum(self.column_widths) == 0):
        return
    self._partitions = np.array([[self._partitions[i][j] for j in range(len(self._partitions[i])) if j < len(self.column_widths) and self.column_widths[j] != 0] for i in range(len(self._partitions)) if i < len(self.row_lengths) and self.row_lengths[i] != 0])
    new_col_widths = [w for w in self.column_widths if w != 0]
    new_row_lengths = [r for r in self.row_lengths if r != 0]
    if new_col_widths != self.column_widths:
        self.set_columns_cache(self.copy_columns_cache(copy_lengths=False))
    if new_row_lengths != self.row_lengths:
        self.set_index_cache(self.copy_index_cache(copy_lengths=False))
    self._column_widths_cache = new_col_widths
    self._row_lengths_cache = new_row_lengths