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
class SimpleListWalker(MonitoredList[_T], ListWalker):

    def __init__(self, contents: Iterable[_T], wrap_around: bool=False) -> None:
        """
        contents -- list to copy into this object

        wrap_around -- if true, jumps to beginning/end of list on move

        This class inherits :class:`MonitoredList` which means
        it can be treated as a list.

        Changes made to this object (when it is treated as a list) are
        detected automatically and will cause ListBox objects using
        this list walker to be updated.
        """
        if not isinstance(contents, Iterable):
            raise ListWalkerError(f'SimpleListWalker expecting list like object, got: {contents!r}')
        MonitoredList.__init__(self, contents)
        self.focus = 0
        self.wrap_around = wrap_around

    @property
    def contents(self) -> Self:
        """
        Return self.

        Provides compatibility with old SimpleListWalker class.
        """
        return self

    def _get_contents(self) -> Self:
        warnings.warn(f'Method `{self.__class__.__name__}._get_contents` is deprecated, please use property`{self.__class__.__name__}.contents`', DeprecationWarning, stacklevel=3)
        return self

    def _modified(self) -> None:
        if self.focus >= len(self):
            self.focus = max(0, len(self) - 1)
        ListWalker._modified(self)

    def set_modified_callback(self, callback: Callable[[], typing.Any]) -> typing.NoReturn:
        """
        This function inherited from MonitoredList is not implemented in SimpleListWalker.

        Use connect_signal(list_walker, "modified", ...) instead.
        """
        raise NotImplementedError('Use connect_signal(list_walker, "modified", ...) instead.')

    def set_focus(self, position: int) -> None:
        """Set focus position."""
        if not 0 <= position < len(self):
            raise IndexError(f'No widget at position {position}')
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