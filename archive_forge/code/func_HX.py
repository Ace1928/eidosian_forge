from __future__ import annotations
import typing
from dataclasses import dataclass
from functools import cached_property
from .._colormaps import PaletteInterpolatedMap
from .._colormaps._colormap import ColorMapKind
def HX(n: int) -> str:
    """
    Conver 8-Bit int to two character HEX (uppercase)

    Should be in the range [0, 255]
    """
    return f'{hex(n)[2:]:>02}'.upper()