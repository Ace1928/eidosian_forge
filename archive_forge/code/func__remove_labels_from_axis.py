from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def _remove_labels_from_axis(axis: Axis) -> None:
    for t in axis.get_majorticklabels():
        t.set_visible(False)
    if isinstance(axis.get_minor_locator(), ticker.NullLocator):
        axis.set_minor_locator(ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), ticker.NullFormatter):
        axis.set_minor_formatter(ticker.FormatStrFormatter(''))
    for t in axis.get_minorticklabels():
        t.set_visible(False)
    axis.get_label().set_visible(False)