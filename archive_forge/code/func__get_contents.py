from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
def _get_contents(self) -> Self:
    warnings.warn(f'Method `{self.__class__.__name__}._get_contents` is deprecated, please use property`{self.__class__.__name__}.contents`', DeprecationWarning, stacklevel=3)
    return self