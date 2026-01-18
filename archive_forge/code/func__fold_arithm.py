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
def _fold_arithm(self, op) -> Union['OpExpr', LiteralExpr]:
    """
        Fold arithmetic expressions.

        Parameters
        ----------
        op : str

        Returns
        -------
        OpExpr or LiteralExpr
        """
    operands = self.operands
    i = 0
    while i < len(operands):
        if isinstance((o := operands[i]), OpExpr):
            if self.op == o.op:
                operands[i:i + 1] = o.operands
            else:
                i += 1
                continue
        if i == 0:
            i += 1
            continue
        if isinstance(o, LiteralExpr) and isinstance(operands[i - 1], LiteralExpr):
            val = getattr(operands[i - 1].val, op)(o.val)
            operands[i - 1] = LiteralExpr(val).cast(o._dtype)
            del operands[i]
        else:
            i += 1
    return operands[0] if len(operands) == 1 else self