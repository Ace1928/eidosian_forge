from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def _update_pref_col_from_focus(self, w_size: tuple[()] | tuple[int] | tuple[int, int]) -> None:
    """Update self.pref_col from the focus widget."""
    if not hasattr(self.focus, 'get_pref_col'):
        return
    pref_col = self.focus.get_pref_col(w_size)
    if pref_col is not None:
        self.pref_col = pref_col