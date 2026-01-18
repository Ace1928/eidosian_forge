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
class LiteralExpr(BaseExpr):
    """
    An expression tree node to represent a literal value.

    Parameters
    ----------
    val : int, np.int, float, bool, str, np.datetime64 or None
        Literal value.
    dtype : None or dtype, default: None
        Value dtype.

    Attributes
    ----------
    val : int, np.int, float, bool, str, np.datetime64 or None
        Literal value.
    _dtype : dtype
        Literal data type.
    """

    def __init__(self, val, dtype=None):
        if val is not None and (not isinstance(val, (int, float, bool, str, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.datetime64))):
            raise NotImplementedError(f'Literal value {val} of type {type(val)}')
        self.val = val
        if dtype is not None:
            self._dtype = dtype
        elif val is None:
            self._dtype = _get_dtype(float)
        else:
            self._dtype = val.dtype if isinstance(val, np.generic) else _get_dtype(type(val))

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        LiteralExpr
        """
        return LiteralExpr(self.val)

    @_inherit_docstrings(BaseExpr.fold)
    def fold(self):
        return self

    @_inherit_docstrings(BaseExpr.cast)
    def cast(self, res_type):
        val = self.val
        if val is not None:
            if isinstance(val, np.generic):
                val = val.astype(res_type)
            elif is_integer_dtype(res_type):
                val = int(val)
            elif is_float_dtype(res_type):
                val = float(val)
            elif is_bool_dtype(res_type):
                val = bool(val)
            elif is_string_dtype(res_type):
                val = str(val)
            else:
                raise TypeError(f"Cannot cast '{val}' to '{res_type}'")
        return LiteralExpr(val, res_type)

    @_inherit_docstrings(BaseExpr.is_null)
    def is_null(self):
        return LiteralExpr(pandas.isnull(self.val), np.dtype(bool))

    @_inherit_docstrings(BaseExpr.is_null)
    def is_not_null(self):
        return LiteralExpr(not pandas.isnull(self.val), np.dtype(bool))

    @_inherit_docstrings(BaseExpr.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return True

    @_inherit_docstrings(BaseExpr.execute_arrow)
    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        return pa.chunked_array([[self.val] * len(table)], to_arrow_type(self._dtype))

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f'{self.val}[{self._dtype}]'

    def __eq__(self, obj):
        """
        Check if `obj` is a `LiteralExpr` with an equal value.

        Parameters
        ----------
        obj : Any object

        Returns
        -------
        bool
        """
        return isinstance(obj, LiteralExpr) and self.val == obj.val