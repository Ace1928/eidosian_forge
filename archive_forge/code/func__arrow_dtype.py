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
def _arrow_dtype(self) -> pa.DataType:
    """
        Get column's dtype representation in underlying PyArrow table.

        Returns
        -------
        pyarrow.DataType
        """
    return self._pyarrow_table.column(-1).type