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
def _dateFormatter(self, v):
    """Create a formatted label for some value."""
    if not isinstance(v, normalDate.NormalDate):
        v = self._scalar2ND(v)
    d, m = (normalDate._dayOfWeekName, normalDate._monthName)
    try:
        normalDate._dayOfWeekName, normalDate._monthName = (self.dayOfWeekName, self.monthName)
        return v.formatMS(self.xLabelFormat)
    finally:
        normalDate._dayOfWeekName, normalDate._monthName = (d, m)