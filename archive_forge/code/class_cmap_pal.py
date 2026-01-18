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
class cmap_pal(_continuous_color_pal):
    """
    Create a continuous palette using a colormap

    Parameters
    ----------
    name : str
        Name of colormap

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
    >>> palette = cmap_pal('viridis')
    >>> palette([.1, .2, .3, .4, .5])
    ['#482475', '#414487', '#355f8d', '#2a788e', '#21918c']
    """
    name: str

    def __post_init__(self):
        self.cm = get_colormap(self.name)

    def __call__(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
        return self.cm.continuous_palette(x)