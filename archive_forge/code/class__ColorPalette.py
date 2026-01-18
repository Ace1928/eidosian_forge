import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""

    def __enter__(self):
        """Open the context."""
        from .rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        from .rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """Return a color palette with hex codes instead of RGB values."""
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return _ColorPalette(hex)

    def _repr_html_(self):
        """Rich display of the color palette in an HTML frontend."""
        s = 55
        n = len(self)
        html = f'<svg  width="{n * s}" height="{s}">'
        for i, c in enumerate(self.as_hex()):
            html += f'<rect x="{i * s}" y="0" width="{s}" height="{s}" style="fill:{c};stroke-width:2;stroke:rgb(255,255,255)"/>'
        html += '</svg>'
        return html