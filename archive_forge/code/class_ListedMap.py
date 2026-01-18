from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import hex_to_rgb, rgb_to_hex
from ._colormap import ColorMap, ColorMapKind
@dataclass
class ListedMap(ColorMap):
    colors: Sequence[RGBHexColor] | Sequence[RGBColor] | RGBColorArray
    kind: ColorMapKind = ColorMapKind.miscellaneous

    def __post_init__(self):
        from ..named_colors import get_named_color
        self.values = np.linspace(0, 1, len(self.colors))
        if isinstance(self.colors[0], str):
            colors = [hex_to_rgb(get_named_color(c)) for c in self.colors]
        else:
            colors = self.colors
        self.n = len(colors)
        self._data = np.asarray(colors)

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
        idx = (x * self.n).astype(int)
        idx[idx < 0] = 0
        idx[idx >= self.n] = self.n - 1
        arr = self._data.take(idx, axis=0, mode='clip')
        return [rgb_to_hex(c) for c in arr]