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
class FrameMapper:
    """
    A helper class for ``InputMapper``.

    This class is used to map column references to another frame.
    This mapper is used to replace input frame in expressions.

    Parameters
    ----------
    frame : HdkOnNativeDataframe
        Target frame.

    Attributes
    ----------
    _frame : HdkOnNativeDataframe
        Target frame.
    """

    def __init__(self, frame):
        self._frame = frame

    def translate(self, col):
        """
        Translate column reference by its name.

        Parameters
        ----------
        col : str
            A name of the column to translate.

        Returns
        -------
        BaseExpr
            Translated expression.
        """
        return self._frame.ref(col)