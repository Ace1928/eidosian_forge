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
def _set_cells(self, widgets: Sequence[Widget]):
    warnings.warn('only for backwards compatibility.You should use the new standard container property `contents` to modify GridFlow', DeprecationWarning, stacklevel=3)
    self.cells = widgets