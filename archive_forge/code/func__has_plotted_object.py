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
@staticmethod
def _has_plotted_object(ax: Axes) -> bool:
    """check whether ax has data"""
    return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0