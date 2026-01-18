from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_transform(self, op):
    """
        Translate ``TransformNode`` node.

        Parameters
        ----------
        op : TransformNode
            An operation to translate.
        """
    fields = list(op.exprs.keys())
    exprs = self._translate(op.exprs.values())
    self._push(CalciteProjectionNode(fields, exprs))