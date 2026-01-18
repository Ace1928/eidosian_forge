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
def _cmp_op(self, other, op_name):
    """
        Build a comparison expression.

        Parameters
        ----------
        other : BaseExpr
            A value to compare with.
        op_name : str
            The comparison operation name.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
    lhs_dtype_class = self._get_dtype_cmp_class(self._dtype)
    rhs_dtype_class = self._get_dtype_cmp_class(other._dtype)
    res_dtype = _get_dtype(bool)
    if lhs_dtype_class != rhs_dtype_class:
        if op_name == 'eq' or op_name == 'ne':
            return LiteralExpr(op_name == 'ne')
        else:
            raise TypeError(f'Invalid comparison between {self._dtype} and {other._dtype}')
    else:
        cmp = OpExpr(self.binary_operations[op_name], [self, other], res_dtype)
        return build_if_then_else(self.is_null(), LiteralExpr(op_name == 'ne'), cmp, res_dtype)