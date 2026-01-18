from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class DynamicStyleTransformation(StyleTransformation):
    """
    StyleTransformation class that can dynamically returns any
    `StyleTransformation`.

    :param get_style_transformation: Callable that returns a
        :class:`.StyleTransformation` instance.
    """

    def __init__(self, get_style_transformation: Callable[[], StyleTransformation | None]) -> None:
        self.get_style_transformation = get_style_transformation

    def transform_attrs(self, attrs: Attrs) -> Attrs:
        style_transformation = self.get_style_transformation() or DummyStyleTransformation()
        return style_transformation.transform_attrs(attrs)

    def invalidation_hash(self) -> Hashable:
        style_transformation = self.get_style_transformation() or DummyStyleTransformation()
        return style_transformation.invalidation_hash()