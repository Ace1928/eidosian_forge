from math import ceil
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
from modin.utils import _inherit_docstrings
from .buffer import HdkProtocolBuffer
from .utils import arrow_dtype_to_arrow_c, arrow_types_map
@property
def _pandas_dtype(self) -> np.dtype:
    """
        Get column's dtype representation in Modin DataFrame.

        Returns
        -------
        numpy.dtype
        """
    return self._col._df.dtypes.iloc[-1]