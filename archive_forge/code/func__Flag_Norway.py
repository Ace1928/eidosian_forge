from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Norway(self):
    s = _size
    g = Group()
    self._width = s * 1.4
    box = Rect(0, 0, self._width, s, fillColor=colors.red, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    box = Rect(0, 0, self._width, s, fillColor=colors.red, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    whiteline1 = Rect(s * 0.2 * 2, 0, width=s * 0.2, height=s, fillColor=colors.ghostwhite, strokeColor=None, strokeWidth=0)
    g.add(whiteline1)
    whiteline2 = Rect(0, s * 0.4, width=self._width, height=s * 0.2, fillColor=colors.ghostwhite, strokeColor=None, strokeWidth=0)
    g.add(whiteline2)
    blueline1 = Rect(s * 0.225 * 2, 0, width=0.1 * s, height=s, fillColor=colors.darkblue, strokeColor=None, strokeWidth=0)
    g.add(blueline1)
    blueline2 = Rect(0, s * 0.45, width=self._width, height=s * 0.1, fillColor=colors.darkblue, strokeColor=None, strokeWidth=0)
    g.add(blueline2)
    return g