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
def _join_arrow_columns(self, other_modin_frames):
    """
        Join arrow table columns.

        If all the frames have a trivial index and an arrow
        table in partitions, concatenate the table columns.

        Parameters
        ----------
        other_modin_frames : list of HdkOnNativeDataframe
            Frames to join with.

        Returns
        -------
        HdkOnNativeDataframe or None
        """
    frames = [self] + other_modin_frames
    if all((f._index_cols is None and isinstance(f._execute(), (DbTable, pyarrow.Table)) for f in frames)):
        tables = [f._partitions[0][0].get(to_arrow=True) for f in frames]
        column_names = [c for t in tables for c in t.column_names]
        if len(column_names) != len(set(column_names)):
            raise NotImplementedError('Duplicate column names')
        max_len = max((len(t) for t in tables))
        columns = [c for t in tables for c in t.columns]
        for i, col in enumerate(columns):
            if len(col) < max_len:
                columns[i] = pyarrow.chunked_array(col.chunks + [pyarrow.nulls(max_len - len(col), col.type)])
        return self.from_arrow(at=pyarrow.table(columns, column_names), columns=[c for f in frames for c in f.columns], encode_col_names=False)
    return None