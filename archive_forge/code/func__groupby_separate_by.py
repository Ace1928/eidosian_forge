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
def _groupby_separate_by(self, by, drop):
    """
        Separate internal and external groupers in `by` argument of groupby.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list
        drop : bool
            Indicates whether or not by data came from self frame.
            True, by data came from self. False, external by data.

        Returns
        -------
        external_by : list of BaseQueryCompiler and arrays
            Values to group by.
        internal_by : list of str
            List of column names from `self` to group by.
        by_positions : list of ints
            Specifies the order of grouping by `internal_by` and `external_by` columns.
            Each element in `by_positions` specifies an index from either `external_by` or `internal_by`.
            Indices for `external_by` are positive and start from 0. Indices for `internal_by` are negative
            and start from -1 (so in order to convert them to a valid indices one should do ``-idx - 1``)
            '''
            by_positions = [0, -1, 1, -2, 2, 3]
            internal_by = ["col1", "col2"]
            external_by = [sr1, sr2, sr3, sr4]

            df.groupby([sr1, "col1", sr2, "col2", sr3, sr4])
            '''.
        """
    if isinstance(by, type(self)):
        if drop:
            internal_by = by.columns.tolist()
            external_by = []
            by_positions = [-i - 1 for i in range(len(internal_by))]
        else:
            internal_by = []
            external_by = [by]
            by_positions = [i for i in range(len(external_by[0].columns))]
    else:
        if not isinstance(by, list):
            by = [by] if by is not None else []
        internal_by = []
        external_by = []
        external_by_counter = 0
        by_positions = []
        for o in by:
            if isinstance(o, pandas.Grouper) and o.key in self.columns:
                internal_by.append(o.key)
                by_positions.append(-len(internal_by))
            elif hashable(o) and o in self.columns:
                internal_by.append(o)
                by_positions.append(-len(internal_by))
            else:
                external_by.append(o)
                for _ in range(len(o.columns) if isinstance(o, type(self)) else 1):
                    by_positions.append(external_by_counter)
                    external_by_counter += 1
    return (external_by, internal_by, by_positions)