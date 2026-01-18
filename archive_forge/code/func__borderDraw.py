from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _borderDraw(self, f):
    s = self.size
    g = Group()
    g.add(f)
    x, y, sW = (self.x + self.dx, self.y + self.dy, self.strokeWidth / 2.0)
    g.insert(0, Rect(-sW, -sW, width=getattr(self, '_width', 2 * s) + 3 * sW, height=getattr(self, '_height', s) + 2 * sW, fillColor=None, strokeColor=self.strokeColor, strokeWidth=sW * 2))
    g.shift(x, y)
    g.scale(s / _size, s / _size)
    return g