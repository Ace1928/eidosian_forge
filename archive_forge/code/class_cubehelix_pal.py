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
class cubehelix_pal(_discrete_pal):
    """
    Utility for creating discrete palette from the cubehelix system.

    This produces a colormap with linearly-decreasing (or increasing)
    brightness. That means that information will be preserved if printed to
    black and white or viewed by someone who is colorblind.

    Parameters
    ----------
    start : float (0 <= start <= 3)
        The hue at the start of the helix.
    rot : float
        Rotations around the hue wheel over the range of the palette.
    gamma : float (0 <= gamma)
        Gamma factor to emphasize darker (gamma < 1) or lighter (gamma > 1)
        colors.
    hue : float (0 <= hue <= 1)
        Saturation of the colors.
    dark : float (0 <= dark <= 1)
        Intensity of the darkest color in the palette.
    light : float (0 <= light <= 1)
        Intensity of the lightest color in the palette.
    reverse : bool
        If True, the palette will go from dark to light.

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        equally spaced colors.


    References
    ----------
    Green, D. A. (2011). "A colour scheme for the display of astronomical
    intensity images". Bulletin of the Astromical Society of India, Vol. 39,
    p. 289-295.

    Examples
    --------
    >>> palette = cubehelix_pal()
    >>> palette(5)
    ['#edd1cb', '#d499a7', '#aa678f', '#6e4071', '#2d1e3e']
    """
    start: int = 0
    rotation: float = 0.4
    gamma: float = 1.0
    hue: float = 0.8
    light: float = 0.85
    dark: float = 0.15
    reverse: bool = False

    def __post_init__(self):
        self._chmap = CubeHelixMap(self.start, self.rotation, self.gamma, self.hue, self.light, self.dark, self.reverse)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        return self._chmap.discrete_palette(n)