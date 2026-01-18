from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Japan(self):
    s = _size
    g = Group()
    w = self._width = s * 1.5
    g.add(Rect(0, 0, w, s, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Circle(cx=w / 2.0, cy=s / 2.0, r=0.3 * w, fillColor=colors.red, strokeColor=None, strokeWidth=0))
    return g