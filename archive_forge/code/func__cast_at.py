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
def _cast_at(self, new_schema: pa.Schema):
    """
        Cast underlying PyArrow table with the passed schema.

        Parameters
        ----------
        new_schema : pyarrow.Schema
            New schema to cast the table.

        Notes
        -----
        This method modifies the column inplace by replacing the wrapped ``HdkProtocolDataframe``
        with the new one holding the casted PyArrow table.
        """
    casted_at = self._pyarrow_table.cast(new_schema)
    self._col = type(self._col)(self._col._df.from_arrow(casted_at), self._col._nan_as_null, self._col._allow_copy)