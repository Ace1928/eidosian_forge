from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _getLit(col, shd=None, lighting=0.1):
    if shd is None:
        from reportlab.lib.colors import Whiter
        if col:
            shd = Whiter(col, 1 - lighting)
    return shd