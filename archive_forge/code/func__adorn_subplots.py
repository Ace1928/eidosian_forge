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
def _adorn_subplots(self, fig: Figure) -> None:
    """Common post process unrelated to data"""
    if len(self.axes) > 0:
        all_axes = self._get_subplots(fig)
        nrows, ncols = self._get_axes_layout(fig)
        handle_shared_axes(axarr=all_axes, nplots=len(all_axes), naxes=nrows * ncols, nrows=nrows, ncols=ncols, sharex=self.sharex, sharey=self.sharey)
    for ax in self.axes:
        ax = getattr(ax, 'right_ax', ax)
        if self.yticks is not None:
            ax.set_yticks(self.yticks)
        if self.xticks is not None:
            ax.set_xticks(self.xticks)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylabel is not None:
            ax.set_ylabel(pprint_thing(self.ylabel))
        ax.grid(self.grid)
    if self.title:
        if self.subplots:
            if is_list_like(self.title):
                if len(self.title) != self.nseries:
                    raise ValueError(f'The length of `title` must equal the number of columns if using `title` of type `list` and `subplots=True`.\nlength of title = {len(self.title)}\nnumber of columns = {self.nseries}')
                for ax, title in zip(self.axes, self.title):
                    ax.set_title(title)
            else:
                fig.suptitle(self.title)
        else:
            if is_list_like(self.title):
                msg = 'Using `title` of type `list` is not supported unless `subplots=True` is passed'
                raise ValueError(msg)
            self.axes[0].set_title(self.title)