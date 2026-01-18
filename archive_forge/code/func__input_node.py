from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _input_node(self, idx):
    """
        Get an input calcite node by index.

        Parameters
        ----------
        idx : int
            An input node's index.

        Returns
        -------
        CalciteBaseNode
        """
    return self._input_nodes()[idx]