import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def _patch_colormap_display():
    """Simplify the rich display of matplotlib color maps in a notebook."""

    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        import io
        from PIL import Image
        import numpy as np
        IMAGE_SIZE = (400, 50)
        X = np.tile(np.linspace(0, 1, IMAGE_SIZE[0]), (IMAGE_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        Image.fromarray(pixels).save(png_bytes, format='png')
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        import base64
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        return '<img ' + 'alt="' + self.name + ' color map" ' + 'title="' + self.name + '"' + 'src="data:image/png;base64,' + png_base64 + '">'
    mpl.colors.Colormap._repr_png_ = _repr_png_
    mpl.colors.Colormap._repr_html_ = _repr_html_