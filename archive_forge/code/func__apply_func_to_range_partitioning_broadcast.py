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
def _apply_func_to_range_partitioning_broadcast(self, right, func, key, new_index=None, new_columns=None, new_dtypes=None):
    """
        Apply `func` against two dataframes using range-partitioning implementation.

        The method first builds range-partitioning for both dataframes using the data from
        `self[key]`, after that, it applies `func` row-wise to `self` frame and
        broadcasts row-parts of `right` to `self`.

        Parameters
        ----------
        right : PandasDataframe
        func : callable(left : pandas.DataFrame, right : pandas.DataFrame) -> pandas.DataFrame
        key : list of labels
            Columns to use to build range-partitioning. Must present in both dataframes.
        new_index : pandas.Index, optional
            Index values to write to the result's cache.
        new_columns : pandas.Index, optional
            Column values to write to the result's cache.
        new_dtypes : pandas.Series or ModinDtypes, optional
            Dtype values to write to the result's cache.

        Returns
        -------
        PandasDataframe
        """
    if self._partitions.shape[0] == 1:
        result = self.broadcast_apply_full_axis(axis=1, func=func, new_columns=new_columns, dtypes=new_dtypes, other=right)
        return result
    if not isinstance(key, list):
        key = [key]
    shuffling_functions = ShuffleSortFunctions(self, key, ascending=True, ideal_num_new_partitions=self._partitions.shape[0])
    key_indices = self.columns.get_indexer_for(key)
    partition_indices = np.unique(np.digitize(key_indices, np.cumsum(self.column_widths)))
    new_partitions = self._partition_mgr_cls.shuffle_partitions(self._partitions, partition_indices, shuffling_functions, func, right_partitions=right._partitions)
    return self.__constructor__(new_partitions, index=new_index, columns=new_columns, dtypes=new_dtypes)