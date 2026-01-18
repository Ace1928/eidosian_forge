from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
def _color_to_rgb(self, color: str) -> tuple[float, float, float]:
    """
        Parse `style.Attrs` color into RGB tuple.
        """
    try:
        from prompt_toolkit.output.vt100 import ANSI_COLORS_TO_RGB
        r, g, b = ANSI_COLORS_TO_RGB[color]
        return (r / 255.0, g / 255.0, b / 255.0)
    except KeyError:
        pass
    return (int(color[0:2], 16) / 255.0, int(color[2:4], 16) / 255.0, int(color[4:6], 16) / 255.0)