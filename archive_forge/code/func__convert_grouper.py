from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core import algorithms
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.io.formats.printing import pprint_thing
def _convert_grouper(axis: Index, grouper):
    if isinstance(grouper, dict):
        return grouper.get
    elif isinstance(grouper, Series):
        if grouper.index.equals(axis):
            return grouper._values
        else:
            return grouper.reindex(axis)._values
    elif isinstance(grouper, MultiIndex):
        return grouper._values
    elif isinstance(grouper, (list, tuple, Index, Categorical, np.ndarray)):
        if len(grouper) != len(axis):
            raise ValueError('Grouper and axis must be same length')
        if isinstance(grouper, (list, tuple)):
            grouper = com.asarray_tuplesafe(grouper)
        return grouper
    else:
        return grouper