from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_frame(self, op):
    """
        Translate ``FrameNode`` node.

        Parameters
        ----------
        op : FrameNode
            A frame to translate.
        """
    self._push(CalciteScanNode(op.modin_frame))