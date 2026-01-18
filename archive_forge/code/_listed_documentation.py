from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import hex_to_rgb, rgb_to_hex
from ._colormap import ColorMap, ColorMapKind

        Lookup colors in the interpolated ranges

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        