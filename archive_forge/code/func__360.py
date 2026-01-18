from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _360(a):
    a %= 360
    if a < -1e-06:
        a += 360
    return a