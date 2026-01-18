from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import hex_to_rgb, rgb_to_hex
from ._colormap import ColorMap, ColorMapKind
def _generate_colors(self, x: FloatArrayLike) -> Sequence[RGBHexColor]:
    """
        Lookup colors in the interpolated ranges

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        """
    x = np.asarray(x)
    idx = np.round(x * 255 + ROUNDING_JITTER).astype(int)
    arr = np.column_stack([self._r_lookup[idx], self._g_lookup[idx], self._b_lookup[idx]])
    return [rgb_to_hex(c) for c in arr]