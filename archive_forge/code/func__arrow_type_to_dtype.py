import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@classmethod
def _arrow_type_to_dtype(cls, arrow_type):
    """
        Convert an arrow data type to a pandas data type.

        Parameters
        ----------
        arrow_type : arrow dtype
            Arrow data type to be converted to a pandas data type.

        Returns
        -------
        object
            Any dtype compatible with pandas.
        """
    import pyarrow
    try:
        res = arrow_type.to_pandas_dtype()
    except NotImplementedError:
        if pyarrow.types.is_time(arrow_type):
            res = np.dtype(datetime.time)
        else:
            raise
    if not isinstance(res, (np.dtype, str)):
        return np.dtype(res)
    return res