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
def _prepare_frame_to_broadcast(self, axis, indices, broadcast_all):
    """
        Compute the indices to broadcast `self` considering `indices`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to broadcast along.
        indices : dict
            Dict of indices and internal indices of partitions where `self` must
            be broadcasted.
        broadcast_all : bool
            Whether broadcast the whole axis of `self` frame or just a subset of it.

        Returns
        -------
        dict
            Dictionary with indices of partitions to broadcast.

        Notes
        -----
        New dictionary of indices of `self` partitions represents that
        you want to broadcast `self` at specified another partition named `other`. For example,
        Dictionary {key: {key1: [0, 1], key2: [5]}} means, that in `other`[key] you want to
        broadcast [self[key1], self[key2]] partitions and internal indices for `self` must be [[0, 1], [5]]
        """
    if broadcast_all:
        sizes = self.row_lengths if axis else self.column_widths
        return {key: dict(enumerate(sizes)) for key in indices.keys()}
    passed_len = 0
    result_dict = {}
    for part_num, internal in indices.items():
        result_dict[part_num] = self._get_dict_of_block_index(axis ^ 1, np.arange(passed_len, passed_len + len(internal)))
        passed_len += len(internal)
    return result_dict