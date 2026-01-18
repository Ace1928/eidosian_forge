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
class manual_pal(_discrete_pal):
    """
    Create a palette from a list of values

    Parameters
    ----------
    values : sequence
        Values that will be returned by the palette function.

    Returns
    -------
    out : function
        A function palette that takes a single
        :class:`int` parameter ``n`` and returns ``n`` values.

    Examples
    --------
    >>> palette = manual_pal(['a', 'b', 'c', 'd', 'e'])
    >>> palette(3)
    ['a', 'b', 'c']
    """
    values: Sequence[Any]

    def __post_init__(self):
        self.size = len(self.values)

    def __call__(self, n: int) -> Sequence[Any]:
        if n > self.size:
            warn(f'Palette can return a maximum of {self.size} values. {n} values requested.')
        return self.values[:n]