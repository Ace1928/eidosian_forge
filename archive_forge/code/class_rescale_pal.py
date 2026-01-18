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
class rescale_pal(_continuous_pal):
    """
    Rescale the input to the specific output range.

    Useful for alpha, size, and continuous position.

    Parameters
    ----------
    range : tuple
        Range of the scale

    Returns
    -------
    out : function
        Palette function that takes a sequence of values
        in the range ``[0, 1]`` and returns values in
        the specified range.

    Examples
    --------
    >>> palette = rescale_pal()
    >>> palette([0, .2, .4, .6, .8, 1])
    array([0.1 , 0.28, 0.46, 0.64, 0.82, 1.  ])

    The returned palette expects inputs in the ``[0, 1]``
    range. Any value outside those limits is clipped to
    ``range[0]`` or ``range[1]``.

    >>> palette([-2, -1, 0.2, .4, .8, 2, 3])
    array([0.1 , 0.1 , 0.28, 0.46, 0.82, 1.  , 1.  ])
    """
    range: TupleFloat2 = (0.1, 1)

    def __call__(self, x: FloatArrayLike) -> NDArrayFloat:
        return rescale(x, self.range, _from=(0, 1))