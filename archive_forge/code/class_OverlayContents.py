from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasOverlay, CompositeCanvas
from urwid.split_repr import remove_defaults
from .constants import (
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .padding import calculate_left_right_padding
from .widget import Widget, WidgetError, WidgetWarning
class OverlayContents(typing.MutableSequence[typing.Tuple[typing.Union[TopWidget, BottomWidget], OverlayOptions]]):

    def __len__(inner_self) -> int:
        return 2
    __getitem__ = self._contents__getitem__
    __setitem__ = self._contents__setitem__

    def __delitem__(self, index: int | slice) -> typing.NoReturn:
        raise TypeError('OverlayContents is fixed-sized sequence')

    def insert(self, index: int | slice, value: typing.Any) -> typing.NoReturn:
        raise TypeError('OverlayContents is fixed-sized sequence')

    def __repr__(inner_self) -> str:
        return repr(f'<{inner_self.__class__.__name__}({[inner_self[0], inner_self[1]]})> for {self}')

    def __rich_repr__(inner_self) -> Iterator[tuple[str | None, typing.Any] | typing.Any]:
        for val in inner_self:
            yield (None, val)

    def __iter__(inner_self) -> Iterator[tuple[Widget, OverlayOptions]]:
        for idx in range(2):
            yield inner_self[idx]