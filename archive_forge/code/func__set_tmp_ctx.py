from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _set_tmp_ctx(self, input_frames, input_nodes):
    """
        Create a temporary input context manager.

        This method is deprecated.

        Parameters
        ----------
        input_frames : list of DFAlgNode
            Input nodes of the currently translated node.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Returns
        -------
        InputContextMgr
            Created input context manager.
        """
    return self.InputContextMgr(self, input_frames, input_nodes)