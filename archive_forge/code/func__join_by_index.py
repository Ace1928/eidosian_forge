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
def _join_by_index(self, other_modin_frames, how, sort, ignore_index):
    """
        Perform equi-join operation for multiple frames by index columns.

        Parameters
        ----------
        other_modin_frames : list of HdkOnNativeDataframe
            Frames to join with.
        how : str
            A type of join.
        sort : bool
            Sort the result by join keys.
        ignore_index : bool
            If True then reset column index for the resulting frame.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    try:
        check_join_supported(how)
    except NotImplementedError as err:
        if (frame := self._join_arrow_columns(other_modin_frames)) is not None:
            return frame
        raise err
    lhs = self._maybe_materialize_rowid()
    reset_index_names = False
    new_columns_dtype = self.columns.dtype
    for rhs in other_modin_frames:
        rhs = rhs._maybe_materialize_rowid()
        if len(lhs._index_cols) != len(rhs._index_cols):
            raise NotImplementedError('join by indexes with different sizes is not supported')
        if new_columns_dtype != rhs.columns.dtype:
            new_columns_dtype = None
        reset_index_names = reset_index_names or lhs._index_cols != rhs._index_cols
        condition = lhs._build_equi_join_condition(rhs, lhs._index_cols, rhs._index_cols)
        exprs = lhs._index_exprs()
        new_columns = lhs.columns.to_list()
        for col in lhs.columns:
            exprs[col] = lhs.ref(col)
        for col in rhs.columns:
            new_col_name = col
            rename_idx = 0
            while new_col_name in exprs:
                new_col_name = f'{col}{rename_idx}'
                rename_idx += 1
            exprs[new_col_name] = rhs.ref(col)
            new_columns.append(new_col_name)
        op = JoinNode(lhs, rhs, how=how, exprs=exprs, condition=condition)
        new_columns = Index.__new__(Index, data=new_columns, dtype=new_columns_dtype)
        lhs = lhs.__constructor__(dtypes=lhs._dtypes_for_exprs(exprs), columns=new_columns, index_cols=lhs._index_cols, op=op, force_execution_mode=self._force_execution_mode)
    if sort:
        lhs = lhs.sort_rows(lhs._index_cols, ascending=True, ignore_index=False, na_position='last')
    if reset_index_names:
        lhs = lhs._reset_index_names()
    if ignore_index:
        new_columns = RangeIndex(range(len(lhs.columns)))
        lhs = lhs._set_columns(new_columns)
    return lhs