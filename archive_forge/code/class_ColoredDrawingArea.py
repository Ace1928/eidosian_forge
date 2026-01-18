from __future__ import annotations
from matplotlib.offsetbox import (
from matplotlib.patches import bbox_artist as mbbox_artist
from matplotlib.transforms import Affine2D, Bbox
from .patches import InsideStrokedRectangle
class ColoredDrawingArea(DrawingArea):
    """
    A Drawing Area with a background color
    """

    def __init__(self, width: float, height: float, xdescent=0.0, ydescent=0.0, clip=True, color='none'):
        super().__init__(width, height, xdescent, ydescent, clip=clip)
        self.patch = InsideStrokedRectangle((0, 0), width=width, height=height, facecolor=color, edgecolor='none', linewidth=0, antialiased=False)
        self.add_artist(self.patch)