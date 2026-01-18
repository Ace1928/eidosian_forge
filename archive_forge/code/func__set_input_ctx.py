from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _set_input_ctx(self, op):
    """
        Create input context manager for a node translation.

        Parameters
        ----------
        op : DFAlgNode
            A translated node.

        Returns
        -------
        InputContextMgr
            Created input context manager.
        """
    input_frames = getattr(op, 'input', [])
    input_nodes = [self._to_calcite(x._op) for x in input_frames]
    return self.InputContextMgr(self, input_frames, input_nodes)