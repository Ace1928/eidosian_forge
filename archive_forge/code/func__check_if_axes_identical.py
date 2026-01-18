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
def _check_if_axes_identical(self, other: 'PandasDataframe', axis: int=0) -> bool:
    """
        Check whether indices/partitioning along the specified `axis` are identical when compared with `other`.

        Parameters
        ----------
        other : PandasDataframe
            Dataframe to compare indices/partitioning with.
        axis : int, default: 0

        Returns
        -------
        bool
        """
    if self.has_axis_cache(axis) and other.has_axis_cache(axis):
        self_cache, other_cache = (self._get_axis_cache(axis), other._get_axis_cache(axis))
        equal_indices = self_cache.equals(other_cache)
        if equal_indices:
            equal_lengths = self_cache.compare_partition_lengths_if_possible(other_cache)
            if isinstance(equal_lengths, bool):
                return equal_lengths
            return self._get_axis_lengths(axis) == other._get_axis_lengths(axis)
        return False
    return self.get_axis(axis).equals(other.get_axis(axis)) and self._get_axis_lengths(axis) == other._get_axis_lengths(axis)