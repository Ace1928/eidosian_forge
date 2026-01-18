from __future__ import annotations
from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import algorithms
from pandas.core.apply import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
from pandas.core.groupby.groupby import (
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby
class NamedAgg(NamedTuple):
    """
    Helper for column specific aggregation with control over output column names.

    Subclass of typing.NamedTuple.

    Parameters
    ----------
    column : Hashable
        Column label in the DataFrame to apply aggfunc.
    aggfunc : function or str
        Function to apply to the provided column. If string, the name of a built-in
        pandas function.

    Examples
    --------
    >>> df = pd.DataFrame({"key": [1, 1, 2], "a": [-1, 0, 1], 1: [10, 11, 12]})
    >>> agg_a = pd.NamedAgg(column="a", aggfunc="min")
    >>> agg_1 = pd.NamedAgg(column=1, aggfunc=lambda x: np.mean(x))
    >>> df.groupby("key").agg(result_a=agg_a, result_1=agg_1)
         result_a  result_1
    key
    1          -1      10.5
    2           1      12.0
    """
    column: Hashable
    aggfunc: AggScalar