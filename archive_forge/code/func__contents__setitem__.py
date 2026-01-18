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
def _contents__setitem__(self, key: Literal['body', 'header', 'footer'], value: tuple[BodyWidget | HeaderWidget | FooterWidget, None]) -> None:
    if key not in {'body', 'header', 'footer'}:
        raise KeyError(f'Frame.contents has no key: {key!r}')
    try:
        value_w, value_options = value
        if value_options is not None:
            raise FrameError(f'added content invalid: {value!r}')
    except (ValueError, TypeError) as exc:
        raise FrameError(f'added content invalid: {value!r}').with_traceback(exc.__traceback__) from exc
    if key == 'body':
        self.body = value_w
    elif key == 'footer':
        self.footer = value_w
    else:
        self.header = value_w