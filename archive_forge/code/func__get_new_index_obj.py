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
def _get_new_index_obj(self, positions, sorted_positions, axis: int) -> 'tuple[pandas.Index, slice | npt.NDArray[np.intp]]':
    """
        Find the new Index object for take_2d_positional result.

        Parameters
        ----------
        positions : Sequence[int]
        sorted_positions : Sequence[int]
        axis : int

        Returns
        -------
        pandas.Index
        slice or Sequence[int]
        """
    if axis == 0:
        idx = self.index
    else:
        idx = self.columns
    if is_range_like(positions) and positions.step > 0:
        monotonic_idx = slice(positions.start, positions.stop, positions.step)
    else:
        monotonic_idx = np.asarray(sorted_positions, dtype=np.intp)
    new_idx = idx[monotonic_idx]
    return (new_idx, monotonic_idx)