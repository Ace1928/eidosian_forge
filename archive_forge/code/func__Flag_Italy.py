from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Italy(self):
    s = _size
    g = Group()
    g.add(Rect(0, 0, s * 2, s, fillColor=colors.forestgreen, strokeColor=None, strokeWidth=0))
    g.add(Rect(2 * s / 3.0, 0, width=s * 4 / 3.0, height=s, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Rect(4 * s / 3.0, 0, width=s * 2 / 3.0, height=s, fillColor=colors.red, strokeColor=None, strokeWidth=0))
    return g