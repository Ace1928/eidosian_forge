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
class _isListOfDaysAndMonths(Validator):
    """This accepts and validates lists of strings like "31-Dec" i.e. dates
    of no particular year.  29 Feb is allowed.  These can be used
    for recurring dates.
    """

    def test(self, x):
        if isSeq(x):
            answer = True
            for element in x:
                try:
                    dd, mm = parseDayAndMonth(element)
                except:
                    answer = False
            return answer
        else:
            return False

    def normalize(self, x):
        return x