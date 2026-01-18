from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _segKey(a):
    return (a.x0, a.x1, a.y0, a.y1, a.s, a.i)