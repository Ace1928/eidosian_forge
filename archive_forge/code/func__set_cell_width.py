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
def _set_cell_width(self, width: int) -> None:
    warnings.warn(f'Method `{self.__class__.__name__}._set_cell_width` is deprecated, please use property `{self.__class__.__name__}.cell_width`', DeprecationWarning, stacklevel=3)
    self.cell_width = width