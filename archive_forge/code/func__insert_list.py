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
def _insert_list(self, loc, name, value):
    """
        Insert a list-like value.

        Parameters
        ----------
        loc : int
        name : str
        value : list

        Returns
        -------
        HdkOnNativeDataframe
        """
    ncols = len(self.columns)
    if loc == -1:
        loc = ncols
    if ncols == 0:
        assert loc == 0
        return self._list_to_df(name, value, True)
    if self._partitions and self._partitions[0][0].raw:
        return self._insert_list_col(loc, name, value)
    if loc == 0 or loc == ncols:
        in_idx = 0 if loc == 0 else 1
        if isinstance(self._op, JoinNode) and self._op.by_rowid and self._op.input[in_idx]._partitions and self._op.input[in_idx]._partitions[0][0].raw:
            lhs = self._op.input[0]
            rhs = self._op.input[1]
            if loc == 0:
                lhs = lhs._insert_list(0, name, value)
                dtype = lhs.dtypes[0]
            else:
                rhs = rhs._insert_list(-1, name, value)
                dtype = rhs.dtypes[-1]
        elif loc == 0:
            lhs = self._list_to_df(name, value, False)
            rhs = self
            dtype = lhs.dtypes[0]
        else:
            lhs = self
            rhs = self._list_to_df(name, value, False)
            dtype = rhs.dtypes[0]
    elif isinstance(self._op, JoinNode) and self._op.by_rowid:
        left_len = len(self._op.input[0].columns)
        if loc < left_len:
            lhs = self._op.input[0]._insert_list(loc, name, value)
            rhs = self._op.input[1]
            dtype = lhs.dtypes[loc]
        else:
            lhs = self._op.input[0]
            rhs = self._op.input[1]._insert_list(loc - left_len, name, value)
            dtype = rhs.dtypes[loc]
    else:
        lexprs = self._index_exprs()
        rexprs = {}
        for i, col in enumerate(self.columns):
            (lexprs if i < loc else rexprs)[col] = self.ref(col)
        lhs = self.__constructor__(columns=self.columns[0:loc], dtypes=self._dtypes_for_exprs(lexprs), op=TransformNode(self, lexprs), index=self._index_cache, index_cols=self._index_cols, force_execution_mode=self._force_execution_mode)._insert_list(loc, name, value)
        rhs = self.__constructor__(columns=self.columns[loc:], dtypes=self._dtypes_for_exprs(rexprs), op=TransformNode(self, rexprs), force_execution_mode=self._force_execution_mode)
        dtype = lhs.dtypes[loc]
    op = self._join_by_rowid_op(lhs, rhs)
    return self._insert_list_col(loc, name, value, dtype, op)