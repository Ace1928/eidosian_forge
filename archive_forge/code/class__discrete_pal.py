from __future__ import annotations
import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn
import numpy as np
from ._colors import (
from .bounds import rescale
from .utils import identity
class _discrete_pal(Protocol):
    """
    Discrete palette maker
    """

    def __call__(self, n: int) -> Sequence[Any]:
        """
        Palette method
        """
        ...