from __future__ import annotations
import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn
import numpy as np
from ._colors import (
from .bounds import rescale
from .utils import identity
class desaturate_pal(gradient_n_pal):
    """
    Create a palette that desaturate a color by some proportion

    Parameters
    ----------
    color : color
        html color name, hex, rgb-tuple
    prop : float
        saturation channel of color will be multiplied by
        this value
    reverse : bool
        Whether to reverse the palette.

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        parameter either a :class:`float` or a sequence
        of floats maps those value(s) onto the palette
        and returns color(s). The float(s) must be
        in the range [0, 1].

    Examples
    --------
    >>> palette = desaturate_pal('red', .1)
    >>> palette([0, .25, .5, .75, 1])
    ['#ff0000', '#e21d1d', '#c53a3a', '#a95656', '#8c7373']
    """

    def __init__(self, color: str, prop: float, reverse: bool=False):
        if not 0 <= prop <= 1:
            raise ValueError('prop must be between 0 and 1')
        if isinstance(color, str):
            color = get_named_color(color)
        rgb = hex_to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        s *= prop
        desaturated_color = rgb_to_hex(colorsys.hls_to_rgb(h, l, s))
        colors = [color, desaturated_color]
        if reverse:
            colors = colors[::-1]
        super().__init__(colors)