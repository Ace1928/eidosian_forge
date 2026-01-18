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
class area_pal(_continuous_pal):
    """
    Point area palette (continuous).

    Parameters
    ----------
    range : tuple
        Numeric vector of length two, giving range of possible sizes.
        Should be greater than 0.

    Returns
    -------
    out : function
        Palette function that takes a sequence of values
        in the range ``[0, 1]`` and returns values in
        the specified range.

    Examples
    --------
    >>> x = np.arange(0, .6, .1)**2
    >>> palette = area_pal()
    >>> palette(x)
    array([1. , 1.5, 2. , 2.5, 3. , 3.5])

    The results are equidistant because the input ``x`` is in
    area space, i.e it is squared.
    """
    range: TupleFloat2 = (1, 6)

    def __call__(self, x: FloatArrayLike) -> NDArrayFloat:
        return rescale(np.sqrt(x), to=self.range, _from=(0, 1))