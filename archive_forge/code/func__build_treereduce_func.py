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
def _build_treereduce_func(self, axis, func):
    """
        Properly formats a TreeReduce result so that the partitioning is correct.

        Parameters
        ----------
        axis : int
            The axis along which to apply the function.
        func : callable
            The function to apply.

        Returns
        -------
        callable
            A function to be shipped to the partitions to be executed.

        Notes
        -----
        This should be used for any TreeReduce style operation that results in a
        reduced data dimensionality (dataframe -> series).
        """

    def _tree_reduce_func(df, *args, **kwargs):
        """Tree-reducer function itself executing `func`, presenting the resulting pandas.Series as pandas.DataFrame."""
        series_result = func(df, *args, **kwargs)
        if axis == 0 and isinstance(series_result, pandas.Series):
            result = pandas.DataFrame(series_result).T
            result.index = [MODIN_UNNAMED_SERIES_LABEL]
        else:
            result = pandas.DataFrame(series_result)
            if isinstance(series_result, pandas.Series):
                result.columns = [MODIN_UNNAMED_SERIES_LABEL]
        return result
    return _tree_reduce_func