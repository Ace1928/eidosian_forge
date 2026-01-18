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
class InputRefExpr(BaseExpr):
    """
    An expression tree node to represent an input frame column.

    Parameters
    ----------
    frame : HdkOnNativeDataframe
        An input frame.
    col : str
        An input column name.
    dtype : dtype
        Input column data type.

    Attributes
    ----------
    modin_frame : HdkOnNativeDataframe
        An input frame.
    column : str
        An input column name.
    _dtype : dtype
        Input column data type.
    """

    def __init__(self, frame, col, dtype):
        self.modin_frame = frame
        self.column = col
        self._dtype = dtype

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        InputRefExpr
        """
        return InputRefExpr(self.modin_frame, self.column, self._dtype)

    def collect_frames(self, frames):
        """
        Add referenced frame to the `frames` set.

        Parameters
        ----------
        frames : set
            Output set of collected frames.
        """
        frames.add(self.modin_frame)

    def _translate_input(self, mapper):
        """
        Translate the referenced column using `mapper`.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input column translation.

        Returns
        -------
        BaseExpr
            The translated expression.
        """
        return mapper.translate(self)

    @_inherit_docstrings(BaseExpr.fold)
    def fold(self):
        return self

    @_inherit_docstrings(BaseExpr.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return True

    @_inherit_docstrings(BaseExpr.execute_arrow)
    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        if self.column == ColNameCodec.ROWID_COL_NAME:
            return pa.chunked_array([range(len(table))], pa.int64())
        return table.column(ColNameCodec.encode(self.column))

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f'{self.modin_frame.id_str()}.{self.column}[{self._dtype}]'