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
def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes:
    if not self.on_right(axes_num):
        return self._get_ax_layer(ax)
    if hasattr(ax, 'right_ax'):
        return ax.right_ax
    elif hasattr(ax, 'left_ax'):
        return ax
    else:
        orig_ax, new_ax = (ax, ax.twinx())
        new_ax._get_lines = orig_ax._get_lines
        new_ax._get_patches_for_fill = orig_ax._get_patches_for_fill
        orig_ax.right_ax, new_ax.left_ax = (new_ax, orig_ax)
        if not self._has_plotted_object(orig_ax):
            orig_ax.get_yaxis().set_visible(False)
        if self.logy is True or self.loglog is True:
            new_ax.set_yscale('log')
        elif self.logy == 'sym' or self.loglog == 'sym':
            new_ax.set_yscale('symlog')
        return new_ax