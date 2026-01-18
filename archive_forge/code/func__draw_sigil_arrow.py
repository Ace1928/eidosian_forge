from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_sigil_arrow(self, bottom, center, top, startangle, endangle, strand, **kwargs):
    """Draw ARROW sigil (PRIVATE)."""
    if strand == 1:
        inner_radius = center
        outer_radius = top
        orientation = 'right'
    elif strand == -1:
        inner_radius = bottom
        outer_radius = center
        orientation = 'left'
    else:
        inner_radius = bottom
        outer_radius = top
        orientation = 'right'
    return self._draw_arc_arrow(inner_radius, outer_radius, startangle, endangle, orientation=orientation, **kwargs)