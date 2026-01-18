from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_filter(self, op):
    """
        Translate ``FilterNode`` node.

        Parameters
        ----------
        op : FilterNode
            An operation to translate.
        """
    condition = self._translate(op.condition)
    self._push(CalciteFilterNode(condition))
    if isinstance(self._input_node(0), CalciteScanNode):
        self._add_projection(op.input[0])