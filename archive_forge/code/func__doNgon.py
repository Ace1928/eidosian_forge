from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def _doNgon(self, n):
    P = []
    size = float(self.size) / 2
    for i in range(n):
        r = (2.0 * i / n + 0.5) * pi
        P.append(size * cos(r))
        P.append(size * sin(r))
    return self._doPolygon(P)