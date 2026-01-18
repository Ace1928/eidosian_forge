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
class LogAxisTickLabellerS(TickLabeller):
    """simple log axis labeller tries to use integers
    and short forms else exponential format"""

    def __call__(self, axis, value):
        e = math_log10(value)
        p = int(e - 0.001 if e < 0 else e + 0.001)
        if p == 0:
            return '1'
        s = '1' + p * '0' if p > 0 else '0.' + -(1 + p) * '0' + '1'
        se = '%.0e' % value
        return se if len(se) < len(s) else s