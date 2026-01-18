from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
class SimplePatchShadow(AbstractPathEffect):
    """A simple shadow via a filled patch."""

    def __init__(self, offset=(2, -2), shadow_rgbFace=None, alpha=None, rho=0.3, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (2, -2)
            The (x, y) offset of the shadow in points.
        shadow_rgbFace : color
            The shadow color.
        alpha : float, default: 0.3
            The alpha transparency of the created shadow patch.
        rho : float, default: 0.3
            A scale factor to apply to the rgbFace color if *shadow_rgbFace*
            is not specified.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.

        """
        super().__init__(offset)
        if shadow_rgbFace is None:
            self._shadow_rgbFace = shadow_rgbFace
        else:
            self._shadow_rgbFace = mcolors.to_rgba(shadow_rgbFace)
        if alpha is None:
            alpha = 0.3
        self._alpha = alpha
        self._rho = rho
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        Overrides the standard draw_path to add the shadow offset and
        necessary color changes for the shadow.
        """
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        if self._shadow_rgbFace is None:
            r, g, b = (rgbFace or (1.0, 1.0, 1.0))[:3]
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_rgbFace
        gc0.set_foreground('none')
        gc0.set_alpha(self._alpha)
        gc0.set_linewidth(0)
        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine + self._offset_transform(renderer), shadow_rgbFace)
        gc0.restore()