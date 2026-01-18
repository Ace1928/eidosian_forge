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
def _set_focus_from_display_widget(self) -> None:
    """
        Set the focus to the item in focus in the display widget.
        """
    pile_focus = self._w.focus
    if not pile_focus:
        return
    c = pile_focus.base_widget
    if c.focus:
        col_focus_position = c.focus_position
    else:
        col_focus_position = 0
    self.focus_position = pile_focus.first_position + col_focus_position