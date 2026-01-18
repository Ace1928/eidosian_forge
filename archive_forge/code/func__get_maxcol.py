from __future__ import annotations
import typing
import warnings
from urwid.split_repr import remove_defaults
from .columns import Columns
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .divider import Divider
from .monitored_list import MonitoredFocusList, MonitoredList
from .padding import Padding
from .pile import Pile
from .widget import Widget, WidgetError, WidgetWarning, WidgetWrap
def _get_maxcol(self, size: tuple[int] | tuple[()]) -> int:
    if size:
        maxcol, = size
        if maxcol < self.cell_width:
            warnings.warn(f'Size is smaller than cell width ({maxcol!r} < {self.cell_width!r})', GridFlowWarning, stacklevel=3)
    else:
        maxcol = len(self) * self.cell_width + (len(self) - 1) * self.h_sep
    return maxcol