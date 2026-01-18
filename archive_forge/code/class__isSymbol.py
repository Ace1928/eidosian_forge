from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
class _isSymbol(Validator):

    def test(self, x):
        return hasattr(x, '__call__') or isinstance(x, Marker) or isinstance(x, _Symbol) or (isClass(x) and issubclass(x, Widget))