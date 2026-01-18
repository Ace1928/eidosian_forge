from __future__ import annotations
from typing import TYPE_CHECKING
from matplotlib import artist
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import _get_textbox  # type: ignore
from matplotlib.transforms import Affine2D
class InsideStrokedRectangle(Rectangle):
    """
    A rectangle whose stroked is fully contained within it
    """

    @artist.allow_rasterization
    def draw(self, renderer):
        """
        Draw with the bounds of the rectangle adjusted to accomodate the stroke
        """
        x, y = self.xy
        w, h = (self.get_width(), self.get_height())
        lw = self.get_linewidth()
        self.set_bounds(x + lw / 2, y + lw / 2, w - lw, h - lw)
        super().draw(renderer)