import re
from typing import Hashable, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow
from pandas._libs.lib import no_default
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import Index, MultiIndex, RangeIndex
from pyarrow.types import is_dictionary
from modin.core.dataframe.base.dataframe.utils import (
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.dataframe.pandas.metadata.dtypes import get_categories_dtype
from modin.core.dataframe.pandas.utils import concatenate
from modin.error_message import ErrorMessage
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from ..db_worker import DbTable
from ..df_algebra import (
from ..expr import (
from ..partitioning.partition_manager import HdkOnNativeDataframePartitionManager
from .utils import (
def _groupby_head_tail(self, agg: str, n: int, cols: Iterable[str]) -> 'HdkOnNativeDataframe':
    """
        Return first/last n rows of each group.

        Parameters
        ----------
        agg : {"head", "tail"}
        n : int
            If positive: number of entries to include from start/end of each group.
            If negative: number of entries to exclude from start/end of each group.
        cols : Iterable[str]
            Group by column names.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    if isinstance(self._op, SortNode):
        base = self._op.input[0]
        order_keys = self._op.columns
        ascending = self._op.ascending
        na_pos = self._op.na_position.upper()
        fold = True
    else:
        base = self._maybe_materialize_rowid()
        order_keys = base._index_cols[0:1]
        ascending = [True]
        na_pos = 'FIRST'
        fold = base is self
    if (n < 0) == (agg == 'head'):
        ascending = [not a for a in ascending]
        na_pos = 'FIRST' if na_pos == 'LAST' else 'LAST'
    partition_keys = [base.ref(col) for col in cols]
    order_keys = [base.ref(col) for col in order_keys]
    row_num_name = '__HDK_ROW_NUMBER__'
    row_num_op = OpExpr('ROW_NUMBER', [], _get_dtype(int))
    row_num_op.set_window_opts(partition_keys, order_keys, ascending, na_pos)
    exprs = base._index_exprs()
    exprs.update(((col, base.ref(col)) for col in base.columns))
    exprs[row_num_name] = row_num_op
    transform = base.copy(columns=list(base.columns) + [row_num_name], dtypes=self._dtypes_for_exprs(exprs), op=TransformNode(base, exprs, fold))
    if n < 0:
        cond = transform.ref(row_num_name).ge(-n + 1)
    else:
        cond = transform.ref(row_num_name).le(n)
    filter = transform.copy(op=FilterNode(transform, cond))
    exprs = filter._index_exprs()
    exprs.update(((col, filter.ref(col)) for col in base.columns))
    return base.copy(op=TransformNode(filter, exprs), partitions=None, index=None)