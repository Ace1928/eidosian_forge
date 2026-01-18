from reportlab.graphics.shapes import Drawing, Line, String, Group, Polygon
from reportlab.lib import colors
from ._AbstractDrawer import AbstractDrawer, draw_box, draw_arrow
from ._AbstractDrawer import draw_cut_corner_box, _stroke_and_fill_colors
from ._AbstractDrawer import intermediate_points, angle2trig, deduplicate
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import ceil
def _draw_sigil_octo(self, bottom, center, top, x1, x2, strand, **kwargs):
    """Draw OCTO sigil, a box with the corners cut off (PRIVATE)."""
    if strand == 1:
        y1 = center
        y2 = top
    elif strand == -1:
        y1 = bottom
        y2 = center
    else:
        y1 = bottom
        y2 = top
    return draw_cut_corner_box((x1, y1), (x2, y2), **kwargs)