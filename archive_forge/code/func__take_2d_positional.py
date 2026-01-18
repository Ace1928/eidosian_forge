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
def _take_2d_positional(self, row_positions: Optional[List[int]]=None, col_positions: Optional[List[int]]=None) -> 'PandasDataframe':
    """
        Lazily select columns or rows from given indices.

        Parameters
        ----------
        row_positions : list-like of ints, optional
            The row positions to extract.
        col_positions : list-like of ints, optional
            The column positions to extract.

        Returns
        -------
        PandasDataframe
             A new PandasDataframe from the mask provided.
        """
    indexers = []
    for axis, indexer in enumerate((row_positions, col_positions)):
        if is_range_like(indexer):
            if indexer.step == 1 and len(indexer) == len(self.get_axis(axis)):
                indexer = None
            elif indexer is not None and (not isinstance(indexer, pandas.RangeIndex)):
                indexer = pandas.RangeIndex(indexer.start, indexer.stop, indexer.step)
        else:
            ErrorMessage.catch_bugs_and_request_email(failure_condition=not (indexer is None or is_list_like(indexer)), extra_log='Mask takes only list-like numeric indexers, ' + f'received: {type(indexer)}')
            if isinstance(indexer, list):
                indexer = np.array(indexer, dtype=np.int64)
        indexers.append(indexer)
    row_positions, col_positions = indexers
    if col_positions is None and row_positions is None:
        return self.copy()
    must_sort_row_pos = row_positions is not None and (not np.all(row_positions[1:] >= row_positions[:-1]))
    must_sort_col_pos = col_positions is not None and (not np.all(col_positions[1:] >= col_positions[:-1]))
    if col_positions is None and row_positions is not None:
        all_rows = None
        if self.has_materialized_index:
            all_rows = len(self.index)
        elif self._row_lengths_cache or must_sort_row_pos:
            all_rows = sum(self.row_lengths)
        base_num_cols = 10
        base_ratio = 0.2
        if all_rows and len(row_positions) > 0.9 * all_rows or (must_sort_row_pos and len(row_positions) * base_num_cols >= min(all_rows * len(self.columns) * base_ratio, len(row_positions) * base_num_cols)):
            return self._reorder_labels(row_positions=row_positions, col_positions=col_positions)
    sorted_row_positions = sorted_col_positions = None
    if row_positions is not None:
        if must_sort_row_pos:
            sorted_row_positions = self._get_sorted_positions(row_positions)
        else:
            sorted_row_positions = row_positions
        row_partitions_dict = self._get_dict_of_block_index(0, sorted_row_positions, are_indices_sorted=True)
        new_row_lengths = self._get_new_lengths(row_partitions_dict, axis=0)
        new_index, _ = self._get_new_index_obj(row_positions, sorted_row_positions, axis=0)
    else:
        row_partitions_dict = {i: slice(None) for i in range(len(self._partitions))}
        new_row_lengths = self._row_lengths_cache
        new_index = self.copy_index_cache(copy_lengths=True)
    if col_positions is not None:
        if must_sort_col_pos:
            sorted_col_positions = self._get_sorted_positions(col_positions)
        else:
            sorted_col_positions = col_positions
        col_partitions_dict = self._get_dict_of_block_index(1, sorted_col_positions, are_indices_sorted=True)
        new_col_widths = self._get_new_lengths(col_partitions_dict, axis=1)
        new_columns, monotonic_col_idx = self._get_new_index_obj(col_positions, sorted_col_positions, axis=1)
        ErrorMessage.catch_bugs_and_request_email(failure_condition=sum(new_col_widths) != len(new_columns), extra_log=f'{sum(new_col_widths)} != {len(new_columns)}.\n' + f'{col_positions}\n{self.column_widths}\n{col_partitions_dict}')
        if self.has_materialized_dtypes:
            new_dtypes = self.dtypes.iloc[monotonic_col_idx]
        elif isinstance(self._dtypes, ModinDtypes):
            try:
                new_dtypes = self._dtypes.lazy_get(monotonic_col_idx, numeric_index=True)
            except (ValueError, NotImplementedError):
                new_dtypes = None
        else:
            new_dtypes = None
    else:
        col_partitions_dict = {i: slice(None) for i in range(len(self._partitions.T))}
        new_col_widths = self._column_widths_cache
        new_columns = self.copy_columns_cache(copy_lengths=True)
        new_dtypes = self.copy_dtypes_cache()
    new_partitions = np.array([[self._partitions[row_idx][col_idx].mask(row_internal_indices, col_internal_indices) for col_idx, col_internal_indices in col_partitions_dict.items()] for row_idx, row_internal_indices in row_partitions_dict.items()])
    intermediate = self.__constructor__(new_partitions, new_index, new_columns, new_row_lengths, new_col_widths, new_dtypes)
    return self._maybe_reorder_labels(intermediate, row_positions, col_positions)