from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
def _cyLine3d(self, y, start, end, _3d_dx, _3d_dy):
    y = self._get_line_pos(y)
    x0 = self._x + start
    x1 = self._x + end
    x0, x1 = (min(x0, x1), max(x0, x1))
    y1 = y + _3d_dy
    return PolyLine([x0, y, x0 + _3d_dx, y1, x1 + _3d_dx, y1], strokeLineJoin=1)