from __future__ import annotations
import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn
import numpy as np
from ._colors import (
from .bounds import rescale
from .utils import identity
@dataclass
class hue_pal(_discrete_pal):
    """
    Utility for making hue palettes for color schemes.

    Parameters
    ----------
    h : float
        first hue. In the [0, 1] range
    l : float
        lightness. In the [0, 1] range
    s : float
        saturation. In the [0, 1] range
    color_space : 'hls' | 'husl'
        Color space to use for the palette

    Returns
    -------
    out : function
        A discrete color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        equally spaced colors. Though the palette
        is continuous, since it is varies the hue it
        is good for categorical data. However if ``n``
        is large enough the colors show continuity.

    Examples
    --------
    >>> hue_pal()(5)
    ['#db5f57', '#b9db57', '#57db94', '#5784db', '#c957db']
    >>> hue_pal(color_space='husl')(5)
    ['#e0697e', '#9b9054', '#569d79', '#5b98ab', '#b675d7']
    """
    h: float = 0.01
    l: float = 0.6
    s: float = 0.65
    color_space: Literal['hls', 'husl'] = 'hls'

    def __post_init__(self):
        h, l, s = (self.h, self.l, self.s)
        if not all((0 <= val <= 1 for val in (h, l, s))):
            msg = f'hue_pal expects values to be between 0 and 1. I got h={h!r}, l={l!r}, s={s!r}'
            raise ValueError(msg)
        if self.color_space not in ('hls', 'husl'):
            msg = "color_space should be one of ['hls', 'husl']"
            raise ValueError(msg)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        lookup = {'husl': husl_palette, 'hls': hls_palette}
        palette = lookup[self.color_space]
        colors = palette(n, h=self.h, l=self.l, s=self.s)
        return [hsluv.rgb_to_hex(c) for c in colors]