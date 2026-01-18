from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Turkey(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.red, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    whitecircle = Circle(cx=s * 0.35 * 2, cy=s / 2.0, r=s * 0.3, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whitecircle)
    redcircle = Circle(cx=s * 0.39 * 2, cy=s / 2.0, r=s * 0.24, fillColor=colors.red, strokeColor=None, strokeWidth=0)
    g.add(redcircle)
    ws = Star()
    ws.angle = 15
    ws.size = s / 5.0
    ws.x = s * 0.5 * 2 + ws.size / 2.0
    ws.y = s * 0.5
    ws.fillColor = colors.mintcream
    ws.strokeColor = None
    g.add(ws)
    return g