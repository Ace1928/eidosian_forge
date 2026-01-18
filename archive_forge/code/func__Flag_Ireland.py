from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Ireland(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.forestgreen, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    whitebox = Rect(s * 2.0 / 3.0, 0, width=2.0 * (s * 2.0) / 3.0, height=s, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whitebox)
    orangebox = Rect(2.0 * (s * 2.0) / 3.0, 0, width=s * 2.0 / 3.0, height=s, fillColor=colors.darkorange, strokeColor=None, strokeWidth=0)
    g.add(orangebox)
    return g