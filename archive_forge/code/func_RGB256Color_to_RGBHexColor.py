from __future__ import annotations
import typing
from dataclasses import dataclass
from functools import cached_property
from .._colormaps import PaletteInterpolatedMap
from .._colormaps._colormap import ColorMapKind
def RGB256Color_to_RGBHexColor(color: RGB256Color) -> RGBHexColor:
    """
    Covert 256Color to HexColor
    """
    return '#' + ''.join((HX(i) for i in color))