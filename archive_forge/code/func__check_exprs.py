import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
def _check_exprs(self, attr) -> bool:
    """
        Check if the specified attribute is True for all expressions.

        Parameters
        ----------
        attr : str

        Returns
        -------
        bool
        """
    stack = list(self.exprs.values())
    while stack:
        expr = stack.pop()
        if not getattr(expr, attr)():
            return False
        if isinstance(expr, OpExpr):
            stack.extend(expr.operands)
    return True