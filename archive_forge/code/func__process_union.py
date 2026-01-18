from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_union(self, op):
    """
        Translate ``UnionNode`` node.

        Parameters
        ----------
        op : UnionNode
            An operation to translate.
        """
    self._push(CalciteUnionNode(self._input_ids(), True))