from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Brazil(self):
    s = _size
    g = Group()
    m = s / 14.0
    self._width = w = m * 20

    def addStar(x, y, size, g=g, w=w, s=s, m=m):
        st = Star()
        st.fillColor = colors.mintcream
        st.size = size * m
        st.x = w / 2.0 + x * (0.35 * m)
        st.y = s / 2.0 + y * (0.35 * m)
        g.add(st)
    g.add(Rect(0, 0, w, s, fillColor=colors.green, strokeColor=None, strokeWidth=0))
    g.add(Polygon(points=[1.7 * m, s / 2.0, w / 2.0, s - 1.7 * m, w - 1.7 * m, s / 2.0, w / 2.0, 1.7 * m], fillColor=colors.yellow, strokeColor=None, strokeWidth=0))
    g.add(Circle(cx=w / 2.0, cy=s / 2.0, r=3.5 * m, fillColor=colors.blue, strokeColor=None, strokeWidth=0))
    g.add(Wedge(w / 2.0 - 2 * m, 0, 8.5 * m, 50, 98.1, 8.5 * m, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Wedge(w / 2.0, s / 2.0, 3.501 * m, 156, 352, 3.501 * m, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Wedge(w / 2.0 - 2 * m, 0, 8 * m, 48.1, 100, 8 * m, fillColor=colors.blue, strokeColor=None, strokeWidth=0))
    g.add(Rect(0, 0, w, s / 4.0 + 1.7 * m, fillColor=colors.green, strokeColor=None, strokeWidth=0))
    g.add(Polygon(points=[1.7 * m, s / 2.0, w / 2.0, s / 2.0 - 2 * m, w - 1.7 * m, s / 2.0, w / 2.0, 1.7 * m], fillColor=colors.yellow, strokeColor=None, strokeWidth=0))
    g.add(Wedge(w / 2.0, s / 2.0, 3.502 * m, 166, 342.1, 3.502 * m, fillColor=colors.blue, strokeColor=None, strokeWidth=0))
    addStar(3.2, 3.5, 0.3)
    addStar(-8.5, 1.5, 0.3)
    addStar(-7.5, -3, 0.3)
    addStar(-4, -5.5, 0.3)
    addStar(0, -4.5, 0.3)
    addStar(7, -3.5, 0.3)
    addStar(-3.5, -0.5, 0.25)
    addStar(0, -1.5, 0.25)
    addStar(1, -2.5, 0.25)
    addStar(3, -7, 0.25)
    addStar(5, -6.5, 0.25)
    addStar(6.5, -5, 0.25)
    addStar(7, -4.5, 0.25)
    addStar(-5.5, -3.2, 0.25)
    addStar(-6, -4.2, 0.25)
    addStar(-1, -2.75, 0.2)
    addStar(2, -5.5, 0.2)
    addStar(4, -5.5, 0.2)
    addStar(5, -7.5, 0.2)
    addStar(5, -5.5, 0.2)
    addStar(6, -5.5, 0.2)
    addStar(-8.8, -3.2, 0.2)
    addStar(2.5, 0.5, 0.2)
    addStar(-0.2, -3.2, 0.14)
    addStar(-7.2, -2, 0.14)
    addStar(0, -8, 0.1)
    sTmp = 'ORDEM E PROGRESSO'
    nTmp = len(sTmp)
    delta = 0.850848010347 / nTmp
    radius = 7.9 * m
    centerx = w / 2.0 - 2 * m
    centery = 0
    for i in range(nTmp):
        rad = 2 * pi - i * delta - 4.60766922527
        x = cos(rad) * radius + centerx
        y = sin(rad) * radius + centery
        if i == 6:
            z = 0.35 * m
        else:
            z = 0.45 * m
        g2 = Group(String(x, y, sTmp[i], fontName='Helvetica-Bold', fontSize=z, strokeColor=None, fillColor=colors.green))
        g2.rotate(rad)
        g.add(g2)
    return g