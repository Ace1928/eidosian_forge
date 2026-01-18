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
def _get_c_values(self, color, color_by_categorical: bool, c_is_column: bool):
    c = self.c
    if c is not None and color is not None:
        raise TypeError('Specify exactly one of `c` and `color`')
    if c is None and color is None:
        c_values = self.plt.rcParams['patch.facecolor']
    elif color is not None:
        c_values = color
    elif color_by_categorical:
        c_values = self.data[c].cat.codes
    elif c_is_column:
        c_values = self.data[c].values
    else:
        c_values = c
    return c_values