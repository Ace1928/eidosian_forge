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
class TransformMapper:
    """
    A helper class for ``InputMapper``.

    This class is used to map column references to expressions used
    for their computation. This mapper is used to fold expressions
    from multiple ``TransformNode``-s into a single expression.

    Parameters
    ----------
    op : TransformNode
        Transformation used for mapping.

    Attributes
    ----------
    _op : TransformNode
        Transformation used for mapping.
    """

    def __init__(self, op):
        self._op = op

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
        if col == ColNameCodec.ROWID_COL_NAME:
            return self._op.input[0].ref(col)
        return self._op.exprs[col]