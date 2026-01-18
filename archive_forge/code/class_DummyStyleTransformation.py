from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class DummyStyleTransformation(StyleTransformation):
    """
    Don't transform anything at all.
    """

    def transform_attrs(self, attrs: Attrs) -> Attrs:
        return attrs

    def invalidation_hash(self) -> Hashable:
        return 'dummy-style-transformation'