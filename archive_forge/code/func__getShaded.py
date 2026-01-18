from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _getShaded(col, shd=None, shading=0.1):
    if shd is None:
        from reportlab.lib.colors import Blacker
        if col:
            shd = Blacker(col, 1 - shading)
    return shd