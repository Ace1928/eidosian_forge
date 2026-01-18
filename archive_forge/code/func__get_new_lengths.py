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
def _get_new_lengths(self, partitions_dict, *, axis: int) -> List[int]:
    """
        Find lengths of new partitions.

        Parameters
        ----------
        partitions_dict : dict
        axis : int

        Returns
        -------
        list[int]
        """
    if axis == 0:
        axis_lengths = self.row_lengths
    else:
        axis_lengths = self.column_widths
    new_lengths = [len(range(*part_indexer.indices(axis_lengths[part_idx])) if isinstance(part_indexer, slice) else part_indexer) for part_idx, part_indexer in partitions_dict.items()]
    return new_lengths