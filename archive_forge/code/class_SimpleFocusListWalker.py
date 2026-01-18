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
class SimpleFocusListWalker(ListWalker, MonitoredFocusList[_T]):

    def __init__(self, contents: Iterable[_T], wrap_around: bool=False) -> None:
        """
        contents -- list to copy into this object

        wrap_around -- if true, jumps to beginning/end of list on move

        This class inherits :class:`MonitoredList` which means
        it can be treated as a list.

        Changes made to this object (when it is treated as a list) are
        detected automatically and will cause ListBox objects using
        this list walker to be updated.

        Also, items added or removed before the widget in focus with
        normal list methods will cause the focus to be updated
        intelligently.
        """
        if not isinstance(contents, Iterable):
            raise ListWalkerError(f'SimpleFocusListWalker expecting iterable object, got: {contents!r}')
        MonitoredFocusList.__init__(self, contents)
        self.wrap_around = wrap_around

    def set_modified_callback(self, callback: typing.Any) -> typing.NoReturn:
        """
        This function inherited from MonitoredList is not
        implemented in SimpleFocusListWalker.

        Use connect_signal(list_walker, "modified", ...) instead.
        """
        raise NotImplementedError('Use connect_signal(list_walker, "modified", ...) instead.')

    def set_focus(self, position: int) -> None:
        """Set focus position."""
        self.focus = position
        self._modified()

    def next_position(self, position: int) -> int:
        """
        Return position after start_from.
        """
        if len(self) - 1 <= position:
            if self.wrap_around:
                return 0
            raise IndexError
        return position + 1

    def prev_position(self, position: int) -> int:
        """
        Return position before start_from.
        """
        if position <= 0:
            if self.wrap_around:
                return len(self) - 1
            raise IndexError
        return position - 1

    def positions(self, reverse: bool=False) -> Iterable[int]:
        """
        Optional method for returning an iterable of positions.
        """
        if reverse:
            return range(len(self) - 1, -1, -1)
        return range(len(self))