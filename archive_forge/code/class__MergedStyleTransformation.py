from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class _MergedStyleTransformation(StyleTransformation):

    def __init__(self, style_transformations: Sequence[StyleTransformation]) -> None:
        self.style_transformations = style_transformations

    def transform_attrs(self, attrs: Attrs) -> Attrs:
        for transformation in self.style_transformations:
            attrs = transformation.transform_attrs(attrs)
        return attrs

    def invalidation_hash(self) -> Hashable:
        return tuple((t.invalidation_hash() for t in self.style_transformations))