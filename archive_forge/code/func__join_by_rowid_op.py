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
@staticmethod
def _join_by_rowid_op(lhs, rhs):
    """
        Create a JoinNode for join by rowid.

        Parameters
        ----------
        lhs : HdkOnNativeDataframe
        rhs : HdkOnNativeDataframe

        Returns
        -------
        JoinNode
        """
    exprs = lhs._index_exprs() if lhs._index_cols else rhs._index_exprs()
    exprs.update(((c, lhs.ref(c)) for c in lhs.columns))
    exprs.update(((c, rhs.ref(c)) for c in rhs.columns))
    condition = lhs._build_equi_join_condition(rhs, [ROWID_COL_NAME], [ROWID_COL_NAME])
    return JoinNode(lhs, rhs, exprs=exprs, condition=condition)