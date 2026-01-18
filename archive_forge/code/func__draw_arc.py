from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_arc(self, inner_radius, outer_radius, startangle, endangle, color, border=None, colour=None, **kwargs):
    """Return closed path describing an arc box (PRIVATE).

        Arguments:
         - inner_radius  Float distance of inside of arc from drawing center
         - outer_radius  Float distance of outside of arc from drawing center
         - startangle    Float angle subtended by start of arc at drawing center
           (in radians)
         - endangle      Float angle subtended by end of arc at drawing center
           (in radians)
         - color        colors.Color object for arc (overridden by backwards
           compatible argument with UK spelling, colour).

        Returns a closed path object describing an arced box corresponding to
        the passed values.  For very small angles, a simple four sided
        polygon is used.
        """
    if colour is not None:
        color = colour
    strokecolor, color = _stroke_and_fill_colors(color, border)
    if abs(endangle - startangle) > 0.01:
        p = ArcPath(strokeColor=strokecolor, fillColor=color, strokewidth=0)
        p.addArc(self.xcenter, self.ycenter, inner_radius, 90 - endangle * 180 / pi, 90 - startangle * 180 / pi, moveTo=True)
        p.addArc(self.xcenter, self.ycenter, outer_radius, 90 - endangle * 180 / pi, 90 - startangle * 180 / pi, reverse=True)
        p.closePath()
        return p
    else:
        startcos, startsin = (cos(startangle), sin(startangle))
        endcos, endsin = (cos(endangle), sin(endangle))
        x0, y0 = (self.xcenter, self.ycenter)
        x1, y1 = (x0 + inner_radius * startsin, y0 + inner_radius * startcos)
        x2, y2 = (x0 + inner_radius * endsin, y0 + inner_radius * endcos)
        x3, y3 = (x0 + outer_radius * endsin, y0 + outer_radius * endcos)
        x4, y4 = (x0 + outer_radius * startsin, y0 + outer_radius * startcos)
        return draw_polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], color, border)