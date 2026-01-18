from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def _StarFive(self):
    R = float(self.size) / 2
    r = R * sin(18 * _toradians) / cos(36 * _toradians)
    P = []
    angle = 90
    for i in range(5):
        for radius in (R, r):
            theta = angle * _toradians
            P.append(radius * cos(theta))
            P.append(radius * sin(theta))
            angle = angle + 36
    return self._doPolygon(P)