from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Switzerland(self):
    s = _size
    g = Group()
    self._width = s
    g.add(Rect(0, 0, s, s, fillColor=colors.red, strokeColor=colors.black, strokeWidth=0))
    g.add(Line(s / 2.0, s / 5.5, s / 2, s - s / 5.5, fillColor=colors.mintcream, strokeColor=colors.mintcream, strokeWidth=s / 5.0))
    g.add(Line(s / 5.5, s / 2.0, s - s / 5.5, s / 2.0, fillColor=colors.mintcream, strokeColor=colors.mintcream, strokeWidth=s / 5.0))
    return g