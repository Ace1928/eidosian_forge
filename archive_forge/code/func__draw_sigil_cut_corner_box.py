from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_sigil_cut_corner_box(self, bottom, center, top, startangle, endangle, strand, color, border=None, corner=0.5, **kwargs):
    """Draw OCTO sigil, box with corners cut off (PRIVATE)."""
    if strand == 1:
        inner_radius = center
        outer_radius = top
    elif strand == -1:
        inner_radius = bottom
        outer_radius = center
    else:
        inner_radius = bottom
        outer_radius = top
    strokecolor, color = _stroke_and_fill_colors(color, border)
    startangle, endangle = (min(startangle, endangle), max(startangle, endangle))
    angle = endangle - startangle
    middle_radius = 0.5 * (inner_radius + outer_radius)
    boxheight = outer_radius - inner_radius
    corner_len = min(0.5 * boxheight, 0.5 * boxheight * corner)
    shaft_inner_radius = inner_radius + corner_len
    shaft_outer_radius = outer_radius - corner_len
    cornerangle_delta = max(0.0, min(abs(boxheight) * 0.5 * corner / middle_radius, abs(angle * 0.5)))
    if angle < 0:
        cornerangle_delta *= -1
    startcos, startsin = (cos(startangle), sin(startangle))
    endcos, endsin = (cos(endangle), sin(endangle))
    x0, y0 = (self.xcenter, self.ycenter)
    p = ArcPath(strokeColor=strokecolor, fillColor=color, strokeLineJoin=1, strokewidth=0, **kwargs)
    p.addArc(self.xcenter, self.ycenter, inner_radius, 90 - (endangle - cornerangle_delta) * 180 / pi, 90 - (startangle + cornerangle_delta) * 180 / pi, moveTo=True)
    p.lineTo(x0 + shaft_inner_radius * startsin, y0 + shaft_inner_radius * startcos)
    p.lineTo(x0 + shaft_outer_radius * startsin, y0 + shaft_outer_radius * startcos)
    p.addArc(self.xcenter, self.ycenter, outer_radius, 90 - (endangle - cornerangle_delta) * 180 / pi, 90 - (startangle + cornerangle_delta) * 180 / pi, reverse=True)
    p.lineTo(x0 + shaft_outer_radius * endsin, y0 + shaft_outer_radius * endcos)
    p.lineTo(x0 + shaft_inner_radius * endsin, y0 + shaft_inner_radius * endcos)
    p.closePath()
    return p