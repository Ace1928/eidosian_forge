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
def _build_equi_join_condition(self, rhs, lhs_cols, rhs_cols):
    """
        Build condition for equi-join.

        Parameters
        ----------
        rhs : HdkOnNativeDataframe
            Joined frame.
        lhs_cols : list
            Left frame columns to join by.
        rhs_cols : list
            Right frame columns to join by.

        Returns
        -------
        BaseExpr
        """
    condition = [self.ref(lhs_col).eq(rhs.ref(rhs_col)) for lhs_col, rhs_col in zip(lhs_cols, rhs_cols)]
    condition = condition[0] if len(condition) == 1 else OpExpr('AND', condition, _get_dtype(bool))
    return condition