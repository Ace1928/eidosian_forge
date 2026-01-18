from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def _draw_arc_poly(self, inner_radius, outer_radius, inner_startangle, inner_endangle, outer_startangle, outer_endangle, color, border=None, flip=False, **kwargs):
    """Return polygon path describing an arc."""
    strokecolor, color = _stroke_and_fill_colors(color, border)
    x0, y0 = (self.xcenter, self.ycenter)
    if abs(inner_endangle - outer_startangle) > 0.01 or abs(outer_endangle - inner_startangle) > 0.01 or abs(inner_startangle - outer_startangle) > 0.01 or (abs(outer_startangle - outer_startangle) > 0.01):
        p = ArcPath(strokeColor=strokecolor, fillColor=color, strokeLineJoin=1, strokewidth=0)
        i_start = 90 - inner_startangle * 180 / pi
        i_end = 90 - inner_endangle * 180 / pi
        o_start = 90 - outer_startangle * 180 / pi
        o_end = 90 - outer_endangle * 180 / pi
        p.addArc(x0, y0, inner_radius, i_end, i_start, moveTo=True, reverse=True)
        if flip:
            self._draw_arc_line(p, inner_radius, outer_radius, i_end, o_start)
            p.addArc(x0, y0, outer_radius, o_end, o_start, reverse=True)
            self._draw_arc_line(p, outer_radius, inner_radius, o_end, i_start)
        else:
            self._draw_arc_line(p, inner_radius, outer_radius, i_end, o_end)
            p.addArc(x0, y0, outer_radius, o_end, o_start, reverse=False)
            self._draw_arc_line(p, outer_radius, inner_radius, o_start, i_start)
        p.closePath()
        return p
    else:
        inner_startcos, inner_startsin = (cos(inner_startangle), sin(inner_startangle))
        inner_endcos, inner_endsin = (cos(inner_endangle), sin(inner_endangle))
        outer_startcos, outer_startsin = (cos(outer_startangle), sin(outer_startangle))
        outer_endcos, outer_endsin = (cos(outer_endangle), sin(outer_endangle))
        x1, y1 = (x0 + inner_radius * inner_startsin, y0 + inner_radius * inner_startcos)
        x2, y2 = (x0 + inner_radius * inner_endsin, y0 + inner_radius * inner_endcos)
        x3, y3 = (x0 + outer_radius * outer_endsin, y0 + outer_radius * outer_endcos)
        x4, y4 = (x0 + outer_radius * outer_startsin, y0 + outer_radius * outer_startcos)
        return draw_polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], color, border, strokeLineJoin=1)