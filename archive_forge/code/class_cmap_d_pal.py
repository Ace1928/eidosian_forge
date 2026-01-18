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
class cmap_d_pal(_discrete_pal):
    """
    Create a discrete palette from a colormap

    Parameters
    ----------
    name : str
        Name of colormap

    Returns
    -------
    out : function
        A discrete color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        colors. The maximum value of ``n`` varies
        depending on the parameters.

    Examples
    --------
    >>> palette = cmap_d_pal('viridis')
    >>> palette(5)
    ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    """
    name: str

    def __post_init__(self):
        self.cm = get_colormap(self.name)

    def __call__(self, n: int) -> Sequence[RGBHexColor]:
        return self.cm.discrete_palette(n)