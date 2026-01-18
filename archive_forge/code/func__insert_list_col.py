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
def _insert_list_col(self, idx, name, value, dtype=None, op=None):
    """
        Insert a list-like column.

        Parameters
        ----------
        idx : int
        name : str
        value : list
        dtype : dtype, default: None
        op : DFAlgNode, default: None

        Returns
        -------
        HdkOnNativeDataframe
        """
    cols = self.columns.tolist()
    cols.insert(idx, name)
    has_unsupported_data = self._has_unsupported_data
    if self._index_cols:
        idx += len(self._index_cols)
    if dtype is None:
        part, dtype = self._partitions[0][0].insert(idx, name, value)
        part = np.array([[part]])
        if not has_unsupported_data:
            try:
                ensure_supported_dtype(dtype)
            except NotImplementedError:
                has_unsupported_data = True
    else:
        part = None
    dtypes = self._dtypes.tolist()
    dtypes.insert(idx, dtype)
    return self.copy(partitions=part, columns=cols, dtypes=dtypes, op=op, has_unsupported_data=has_unsupported_data)