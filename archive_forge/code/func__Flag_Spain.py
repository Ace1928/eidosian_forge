from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Spain(self):
    s = _size
    g = Group()
    w = self._width = s * 1.5
    g.add(Rect(0, 0, width=w, height=s, fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Rect(0, s / 4.0, width=w, height=s / 2.0, fillColor=colors.yellow, strokeColor=None, strokeWidth=0))
    return g