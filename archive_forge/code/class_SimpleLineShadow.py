from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
class SimpleLineShadow(AbstractPathEffect):
    """A simple shadow via a line."""

    def __init__(self, offset=(2, -2), shadow_color='k', alpha=0.3, rho=0.3, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (2, -2)
            The (x, y) offset to apply to the path, in points.
        shadow_color : color, default: 'black'
            The shadow color.
            A value of ``None`` takes the original artist's color
            with a scale factor of *rho*.
        alpha : float, default: 0.3
            The alpha transparency of the created shadow patch.
        rho : float, default: 0.3
            A scale factor to apply to the rgbFace color if *shadow_color*
            is ``None``.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.
        """
        super().__init__(offset)
        if shadow_color is None:
            self._shadow_color = shadow_color
        else:
            self._shadow_color = mcolors.to_rgba(shadow_color)
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
        if self._shadow_color is None:
            r, g, b = (gc0.get_foreground() or (1.0, 1.0, 1.0))[:3]
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_color
        gc0.set_foreground(shadow_rgbFace)
        gc0.set_alpha(self._alpha)
        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine + self._offset_transform(renderer))
        gc0.restore()