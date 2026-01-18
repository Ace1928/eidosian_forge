import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
def _op_value(self, op_idx: int, table: pa.Table):
    """
        Get the specified operand value.

        Parameters
        ----------
        op_idx : int
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray or expr.val
        """
    expr = self.operands[op_idx]
    return expr.val if isinstance(expr, LiteralExpr) else expr.execute_arrow(table)