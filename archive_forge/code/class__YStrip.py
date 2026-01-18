from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
class _YStrip:

    def __init__(self, y0, y1, slope, fillColor, fillColorShaded, shading=0.1):
        self.y0 = y0
        self.y1 = y1
        self.slope = slope
        self.fillColor = fillColor
        self.fillColorShaded = _getShaded(fillColor, fillColorShaded, shading)