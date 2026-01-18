from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def _stroke_and_fill_colors(color, border):
    """Deal with  border and fill colors (PRIVATE)."""
    if not isinstance(color, colors.Color):
        raise ValueError(f'Invalid color {color!r}')
    if color == colors.white and border is None:
        strokecolor = colors.black
    elif border is None:
        strokecolor = color
    elif border:
        if not isinstance(border, colors.Color):
            raise ValueError(f'Invalid border color {border!r}')
        strokecolor = border
    else:
        strokecolor = None
    return (strokecolor, color)