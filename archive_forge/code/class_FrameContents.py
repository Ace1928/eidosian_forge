from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasCombine, CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, VAlign
from .container import WidgetContainerMixin
from .filler import Filler
from .widget import Widget, WidgetError
class FrameContents(typing.MutableMapping[str, typing.Tuple[typing.Union[BodyWidget, HeaderWidget, FooterWidget], None]]):
    __slots__ = ()

    def __len__(inner_self) -> int:
        return len(inner_self.keys())
    __getitem__ = self._contents__getitem__
    __setitem__ = self._contents__setitem__
    __delitem__ = self._contents__delitem__

    def __iter__(inner_self) -> Iterator[str]:
        yield from inner_self.keys()

    def __repr__(inner_self) -> str:
        return f'<{inner_self.__class__.__name__}({dict(inner_self)}) for {self}>'

    def __rich_repr__(inner_self) -> Iterator[tuple[str | None, typing.Any] | typing.Any]:
        yield from inner_self.items()