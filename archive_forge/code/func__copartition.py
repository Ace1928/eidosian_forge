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
def _copartition(self, axis, other, how, sort, force_repartition=False, fill_value=None):
    """
        Copartition two Modin DataFrames.

        Perform aligning of partitions, index and partition blocks.

        Parameters
        ----------
        axis : {0, 1}
            Axis to copartition along (0 - rows, 1 - columns).
        other : PandasDataframe
            Other Modin DataFrame(s) to copartition against.
        how : str
            How to manage joining the index object ("left", "right", etc.).
        sort : bool
            Whether sort the joined index or not.
        force_repartition : bool, default: False
            Whether force the repartitioning or not. By default,
            this method will skip repartitioning if it is possible. This is because
            reindexing is extremely inefficient. Because this method is used to
            `join` or `append`, it is vital that the internal indices match.
        fill_value : any, default: None
            Value to use for missing values.

        Returns
        -------
        tuple
            Tuple containing:
                1) 2-d NumPy array of aligned left partitions
                2) list of 2-d NumPy arrays of aligned right partitions
                3) joined index along ``axis``, may be ``ModinIndex`` if not materialized
                4) If materialized, list with sizes of partitions along axis that partitioning
                   was done on, otherwise ``None``. This list will be empty if and only if all
                   the frames are empty.
        """
    if isinstance(other, type(self)):
        other = [other]
    if not force_repartition and all((o._check_if_axes_identical(self, axis) for o in other)):
        return (self._partitions, [o._partitions for o in other], self.copy_axis_cache(axis, copy_lengths=True), self._get_axis_lengths_cache(axis))
    self_index = self.get_axis(axis)
    others_index = [o.get_axis(axis) for o in other]
    joined_index, make_reindexer = self._join_index_objects(axis, [self_index] + others_index, how, sort, fill_value)
    frames = [self] + other
    non_empty_frames_idx = [i for i, o in enumerate(frames) if o._partitions.size != 0]
    if len(non_empty_frames_idx) == 0:
        return (self._partitions, [o._partitions for o in other], joined_index, [])
    base_frame_idx = non_empty_frames_idx[0]
    other_frames = frames[base_frame_idx + 1:]
    base_frame = frames[non_empty_frames_idx[0]]
    base_index = base_frame.get_axis(axis)
    do_reindex_base = not base_index.equals(joined_index)
    do_repartition_base = force_repartition or do_reindex_base
    if do_repartition_base:
        reindexed_base = base_frame._partition_mgr_cls.map_axis_partitions(axis, base_frame._partitions, make_reindexer(do_reindex_base, base_frame_idx))
        if axis:
            base_lengths = [obj.width() for obj in reindexed_base[0]]
        else:
            base_lengths = [obj.length() for obj in reindexed_base.T[0]]
    else:
        reindexed_base = base_frame._partitions
        base_lengths = base_frame.column_widths if axis else base_frame.row_lengths
    others_lengths = [o._get_axis_lengths(axis) for o in other_frames]
    do_reindex_others = [not o.get_axis(axis).equals(joined_index) for o in other_frames]
    do_repartition_others = [None] * len(other_frames)
    for i in range(len(other_frames)):
        do_repartition_others[i] = force_repartition or do_reindex_others[i] or others_lengths[i] != base_lengths
    reindexed_other_list = [None] * len(other_frames)
    for i in range(len(other_frames)):
        if do_repartition_others[i]:
            reindexed_other_list[i] = other_frames[i]._partition_mgr_cls.map_axis_partitions(axis, other_frames[i]._partitions, make_reindexer(do_repartition_others[i], base_frame_idx + 1 + i), lengths=base_lengths)
        else:
            reindexed_other_list[i] = other_frames[i]._partitions
    reindexed_frames = [frames[i]._partitions for i in range(base_frame_idx)] + [reindexed_base] + reindexed_other_list
    return (reindexed_frames[0], reindexed_frames[1:], joined_index, base_lengths)