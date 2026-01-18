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
class grey_pal(_discrete_pal):
    """
    Utility for creating continuous grey scale palette

    Parameters
    ----------
    start : float
        grey value at low end of palette
    end : float
        grey value at high end of palette

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        equally spaced colors.

    Examples
    --------
    >>> palette = grey_pal()
    >>> palette(5)
    ['#333333', '#737373', '#989898', '#b4b4b4', '#cccccc']
    """
    start: float = 0.2
    end: float = 0.8

    def __post_init__(self):
        start, end = (self.start, self.end)
        colors = ((start, start, start), (end, end, end))
        self._cmap = InterpolatedMap(colors)

    def __call__(self, n: int) -> Sequence[RGBHexColor | None]:
        gamma = 2.2
        space = np.linspace(self.start ** gamma, self.end ** gamma, n)
        x = (space ** (1.0 / gamma) - self.start) / (self.end - self.start)
        return self._cmap.continuous_palette(x)