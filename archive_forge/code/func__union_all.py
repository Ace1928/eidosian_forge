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
def _union_all(self, axis, other_modin_frames, join='outer', sort=False, ignore_index=False):
    """
        Concatenate frames' rows.

        Parameters
        ----------
        axis : {0, 1}
            Should be 0.
        other_modin_frames : list of HdkOnNativeDataframe
            Frames to concat.
        join : {"outer", "inner"}, default: "outer"
            How to handle columns with mismatched names.
            "inner" - drop such columns. "outer" - fill
            with NULLs.
        sort : bool, default: False
            Sort unaligned columns for 'outer' join.
        ignore_index : bool, default: False
            Ignore index columns.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
    index_cols = None
    col_name_to_dtype = dict()
    for col in self.columns:
        col_name_to_dtype[col] = self._dtypes[col]
    if join == 'inner':
        for frame in other_modin_frames:
            for col in list(col_name_to_dtype):
                if col not in frame.columns:
                    del col_name_to_dtype[col]
    elif join == 'outer':
        for frame in other_modin_frames:
            for col in frame.columns:
                if col not in col_name_to_dtype:
                    col_name_to_dtype[col] = frame._dtypes[col]
    else:
        raise NotImplementedError(f'Unsupported join type join={join!r}')
    frames = []
    for frame in [self] + other_modin_frames:
        if join == 'inner' or len(frame.columns) != 0 or (frame.has_materialized_index and len(frame.index) != 0) or (not frame.has_materialized_index and frame.index_cols):
            if isinstance(frame._op, UnionNode):
                frames.extend(frame._op.input)
            else:
                frames.append(frame)
    if len(col_name_to_dtype) == 0:
        if len(frames) == 0:
            dtypes = pd.Series()
        elif ignore_index:
            index_cols = [UNNAMED_IDX_COL_NAME]
            dtypes = pd.Series([_get_dtype(int)], index=index_cols)
        else:
            index_names = ColNameCodec.concat_index_names(frames)
            index_cols = list(index_names)
            dtypes = pd.Series(index_names.values(), index=index_cols)
    else:
        for frame in other_modin_frames:
            frame_dtypes = frame._dtypes
            for col in col_name_to_dtype:
                if col in frame_dtypes:
                    col_name_to_dtype[col] = pd.core.dtypes.cast.find_common_type([col_name_to_dtype[col], frame_dtypes[col]])
        if sort:
            col_name_to_dtype = dict(((col, col_name_to_dtype[col]) for col in sorted(col_name_to_dtype)))
        if ignore_index:
            table_col_name_to_dtype = col_name_to_dtype
        else:
            table_col_name_to_dtype = ColNameCodec.concat_index_names(frames)
            index_cols = list(table_col_name_to_dtype)
            table_col_name_to_dtype.update(col_name_to_dtype)
        dtypes = pd.Series(table_col_name_to_dtype.values(), index=table_col_name_to_dtype.keys())
        for i, frame in enumerate(frames):
            frame_dtypes = frame._dtypes.get()
            if len(frame_dtypes) != len(dtypes) or any(frame_dtypes.index != dtypes.index) or any(frame_dtypes.values != dtypes.values):
                exprs = dict()
                uses_rowid = False
                for col in table_col_name_to_dtype:
                    if col in frame_dtypes:
                        expr = frame.ref(col)
                    elif col == UNNAMED_IDX_COL_NAME:
                        if frame._index_cols is not None:
                            assert len(frame._index_cols) == 1
                            expr = frame.ref(frame._index_cols[0])
                        else:
                            uses_rowid = True
                            expr = frame.ref(ROWID_COL_NAME)
                    else:
                        expr = LiteralExpr(None, table_col_name_to_dtype[col])
                    if expr._dtype != table_col_name_to_dtype[col]:
                        expr = expr.cast(table_col_name_to_dtype[col])
                    exprs[col] = expr
                frames[i] = frame.__constructor__(columns=dtypes.index, dtypes=dtypes, uses_rowid=uses_rowid, op=TransformNode(frame, exprs), force_execution_mode=frame._force_execution_mode)
    return self.__constructor__(index_cols=index_cols, columns=col_name_to_dtype.keys(), dtypes=dtypes, op=UnionNode(frames, col_name_to_dtype, ignore_index), force_execution_mode=self._force_execution_mode)