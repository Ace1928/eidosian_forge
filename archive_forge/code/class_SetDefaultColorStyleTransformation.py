from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class SetDefaultColorStyleTransformation(StyleTransformation):
    """
    Set default foreground/background color for output that doesn't specify
    anything. This is useful for overriding the terminal default colors.

    :param fg: Color string or callable that returns a color string for the
        foreground.
    :param bg: Like `fg`, but for the background.
    """

    def __init__(self, fg: str | Callable[[], str], bg: str | Callable[[], str]) -> None:
        self.fg = fg
        self.bg = bg

    def transform_attrs(self, attrs: Attrs) -> Attrs:
        if attrs.bgcolor in ('', 'default'):
            attrs = attrs._replace(bgcolor=parse_color(to_str(self.bg)))
        if attrs.color in ('', 'default'):
            attrs = attrs._replace(color=parse_color(to_str(self.fg)))
        return attrs

    def invalidation_hash(self) -> Hashable:
        return ('set-default-color', to_str(self.fg), to_str(self.bg))