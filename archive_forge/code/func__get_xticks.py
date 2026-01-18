from __future__ import annotations
from abc import (
from collections.abc import (
from typing import (
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
from pandas.plotting._matplotlib.tools import (
@final
def _get_xticks(self):
    index = self.data.index
    is_datetype = index.inferred_type in ('datetime', 'date', 'datetime64', 'time')
    x: list[int] | np.ndarray
    if self.use_index:
        if isinstance(index, ABCPeriodIndex):
            x = index.to_timestamp()._mpl_repr()
        elif is_any_real_numeric_dtype(index.dtype):
            x = index._mpl_repr()
        elif isinstance(index, ABCDatetimeIndex) or is_datetype:
            x = index._mpl_repr()
        else:
            self._need_to_set_index = True
            x = list(range(len(index)))
    else:
        x = list(range(len(index)))
    return x