from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _input_nodes(self):
    """
        Get current input calcite nodes.

        Returns
        -------
        list if CalciteBaseNode
        """
    return self._input_ctx().input_nodes