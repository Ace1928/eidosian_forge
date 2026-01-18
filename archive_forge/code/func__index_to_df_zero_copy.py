import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@staticmethod
def _index_to_df_zero_copy(df: pandas.DataFrame, levels: list[Union[str, int]]) -> pandas.DataFrame:
    """
        Convert index `level` of `df` to a ``pandas.DataFrame``.

        Parameters
        ----------
        df : pandas.DataFrame
        levels : list of labels or ints
            Index level to convert to a dataframe.

        Returns
        -------
        pandas.DataFrame
            The columns in the resulting dataframe use the same data arrays as the index levels
            in the original `df`, so no copies.
        """
    data = {df.index.names[lvl] if isinstance(lvl, int) else lvl: df.index.get_level_values(lvl) for lvl in levels}
    index_data = pandas.DataFrame(data, index=df.index, copy=False)
    return index_data