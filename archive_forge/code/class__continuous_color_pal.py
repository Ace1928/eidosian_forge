from __future__ import annotations
import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn
import numpy as np
from ._colors import (
from .bounds import rescale
from .utils import identity
class _continuous_color_pal(Protocol):
    """
    Continuous color palette maker
    """

    def __call__(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
        """
        Palette method
        """
        ...