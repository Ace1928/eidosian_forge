import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def _list_like_func(self, func, axis, *args, **kwargs):
    """
        Apply passed functions to each row/column.

        Parameters
        ----------
        func : list of callable
            List of functions to apply against each row/column.
        axis : {0, 1}
            Target axis to apply functions along. 0 means apply to columns,
            1 means apply to rows.
        *args : args
            Arguments to pass to the specified functions.
        **kwargs : kwargs
            Arguments to pass to the specified functions.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the results of passed functions.
        """
    new_index = [f if isinstance(f, str) else f.__name__ for f in func] if axis == 0 else self.index
    new_columns = [f if isinstance(f, str) else f.__name__ for f in func] if axis == 1 else self.columns
    func = [wrap_udf_function(f) if callable(f) else f for f in func]
    new_modin_frame = self._modin_frame.apply_full_axis(axis, lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs)), new_index=new_index, new_columns=new_columns)
    return self.__constructor__(new_modin_frame)