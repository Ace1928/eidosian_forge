from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class NotAllowed(_Symbol):
    """This draws a 'forbidden' roundel (as used in the no-smoking sign).

        possible attributes:
        'x', 'y', 'size'

        """
    _attrMap = AttrMap(BASE=_Symbol)

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.strokeColor = colors.red
        self.fillColor = colors.white

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        strokeColor = self.strokeColor
        outerCircle = shapes.Circle(cx=self.x + s / 2, cy=self.y + s / 2, r=s / 2 - s / 10, fillColor=self.fillColor, strokeColor=strokeColor, strokeWidth=s / 10.0)
        g.add(outerCircle)
        centerx = self.x + s
        centery = self.y + s / 2 - s / 6
        radius = s - s / 6
        yradius = radius / 2
        xradius = radius / 2
        startangledegrees = 100
        endangledegrees = -80
        degreedelta = 90
        pointslist = []
        a = pointslist.append
        from math import sin, cos, pi
        degreestoradians = pi / 180.0
        radiansdelta = degreedelta * degreestoradians
        startangle = startangledegrees * degreestoradians
        endangle = endangledegrees * degreestoradians
        while endangle < startangle:
            endangle = endangle + 2 * pi
        angle = startangle
        while angle < endangle:
            x = centerx + cos(angle) * radius
            y = centery + sin(angle) * yradius
            a(x)
            a(y)
            angle = angle + radiansdelta
        crossbar = shapes.PolyLine(pointslist, fillColor=strokeColor, strokeColor=strokeColor, strokeWidth=s / 10.0)
        g.add(crossbar)
        return g