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
def _calcTickPositions(self):
    valueMin = cMin = math_log10(self._valueMin)
    valueMax = cMax = math_log10(self._valueMax)
    rr = self.rangeRound
    if rr:
        if rr in ('both', 'ceiling'):
            i = int(valueMax)
            valueMax = i + 1 if i < valueMax else i
        if rr in ('both', 'floor'):
            i = int(valueMin)
            valueMin = i - 1 if i > valueMin else i
    T = [].append
    tv = int(valueMin)
    if tv < valueMin:
        tv += 1
    n = int(valueMax) - tv + 1
    i = max(int(n / self.maximumTicks), 1)
    if i * n > self.maximumTicks:
        i += 1
    self._powerInc = i
    while True:
        if tv > valueMax:
            break
        if tv >= valueMin:
            T(10 ** tv)
        tv += i
    if valueMin != cMin:
        self._valueMin = 10 ** valueMin
    if valueMax != cMax:
        self._valueMax = 10 ** valueMax
    return T.__self__