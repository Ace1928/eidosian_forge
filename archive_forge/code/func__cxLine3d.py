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
def _cxLine3d(self, x, start, end, _3d_dx, _3d_dy):
    x = self._get_line_pos(x)
    y0 = self._y + start
    y1 = self._y + end
    y0, y1 = (min(y0, y1), max(y0, y1))
    x1 = x + _3d_dx
    return PolyLine([x, y0, x1, y0 + _3d_dy, x1, y1 + _3d_dy], strokeLineJoin=1)