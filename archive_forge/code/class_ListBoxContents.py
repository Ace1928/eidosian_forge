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
class ListBoxContents:
    __getitem__ = self._contents__getitem__
    __len__ = self.__len__

    def __repr__(inner_self) -> str:
        return f'<{inner_self.__class__.__name__} for {self!r} at 0x{id(inner_self):X}>'

    def __call__(inner_self) -> Self:
        warnings.warn('ListBox.contents is a property, not a method', DeprecationWarning, stacklevel=3)
        return inner_self