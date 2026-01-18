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
def _compute_axis_labels_and_lengths(self, axis: int, partitions=None):
    """
        Compute the labels for specific `axis`.

        Parameters
        ----------
        axis : int
            Axis to compute labels along.
        partitions : np.ndarray, optional
            This parameter serves compatibility purpose and must always be ``None``.

        Returns
        -------
        pandas.Index
            Labels for the specified `axis`.
        List of int
            Size of partitions alongside specified `axis`.
        """
    ErrorMessage.catch_bugs_and_request_email(failure_condition=partitions is not None, extra_log="'._compute_axis_labels_and_lengths(partitions)' is not yet supported for HDK backend")
    obj = self._execute()
    if axis == 1:
        cols = self._table_cols
        if self._index_cols is not None:
            cols = cols[len(self._index_cols):]
        return (cols, [len(cols)])
    if self._index_cols is None:
        index = RangeIndex(range(len(obj)))
        return (index, [len(index)])
    if isinstance(obj, DbTable):
        obj = obj.to_arrow()
    if isinstance(obj, pyarrow.Table):
        col_names = obj.column_names[len(self._index_cols):]
        index_at = obj.drop(col_names)
        index_df = index_at.to_pandas()
        index_df.set_index(self._index_cols, inplace=True)
        idx = index_df.index
        idx.rename(demangle_index_names(self._index_cols), inplace=True)
        if isinstance(idx, (pd.DatetimeIndex, pd.TimedeltaIndex)) and len(idx) >= 3:
            idx.freq = pd.infer_freq(idx)
        return (idx, [len(idx)])
    else:
        return (obj.index, [len(obj.index)])