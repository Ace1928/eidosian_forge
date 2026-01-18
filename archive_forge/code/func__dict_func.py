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
def _dict_func(self, func, axis, *args, **kwargs):
    """
        Apply passed functions to the specified rows/columns.

        Parameters
        ----------
        func : dict(label) -> [callable, str]
            Dictionary that maps axis labels to the function to apply against them.
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
    if 'axis' not in kwargs:
        kwargs['axis'] = axis
    func = {k: wrap_udf_function(v) if callable(v) else v for k, v in func.items()}

    def dict_apply_builder(df, internal_indices=[]):
        return pandas.DataFrame(df.apply(func, *args, **kwargs))
    labels = list(func.keys())
    return self.__constructor__(self._modin_frame.apply_full_axis_select_indices(axis, dict_apply_builder, labels, new_index=labels if axis == 1 else None, new_columns=labels if axis == 0 else None, keep_remaining=False))