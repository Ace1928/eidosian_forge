from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
def _interpolate_brightness(self, value: float, min_brightness: float, max_brightness: float) -> float:
    """
        Map the brightness to the (min_brightness..max_brightness) range.
        """
    return min_brightness + (max_brightness - min_brightness) * value