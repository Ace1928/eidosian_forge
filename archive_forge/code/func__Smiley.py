from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def _Smiley(self):
    x, y = (self.x + self.dx, self.y + self.dy)
    d = self.size / 2.0
    s = SmileyFace()
    s.fillColor = self.fillColor
    s.strokeWidth = self.strokeWidth
    s.strokeColor = self.strokeColor
    s.x = x - d
    s.y = y - d
    s.size = d * 2
    return s