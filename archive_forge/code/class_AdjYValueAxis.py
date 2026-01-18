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
class AdjYValueAxis(YValueAxis):
    """A Y-axis applying additional rules.

    Depending on the data and some built-in rules, the axis
    may choose to adjust its range and origin.
    """
    _attrMap = AttrMap(BASE=YValueAxis, leftAxisPercent=AttrMapValue(isBoolean, desc='When true add percent sign to label values.'), leftAxisOrigShiftIPC=AttrMapValue(isNumber, desc='Lowest label shift interval ratio.'), leftAxisOrigShiftMin=AttrMapValue(isNumber, desc='Minimum amount to shift.'), leftAxisSkipLL0=AttrMapValue(EitherOr((isBoolean, isListOfNumbers)), desc='Skip/Keep lowest tick label when true/false.\nOr skiplist'), labelVOffset=AttrMapValue(isNumber, desc='add this to the labels'))

    def __init__(self, **kw):
        YValueAxis.__init__(self, **kw)
        self.requiredRange = 30
        self.leftAxisPercent = 1
        self.leftAxisOrigShiftIPC = 0.15
        self.leftAxisOrigShiftMin = 12
        self.leftAxisSkipLL0 = self.labelVOffset = 0
        self.valueSteps = None

    def _rangeAdjust(self):
        """Adjusts the value range of the axis."""
        from reportlab.graphics.charts.utils import find_good_grid, ticks
        y_min, y_max = (self._valueMin, self._valueMax)
        m = self.maximumTicks
        n = list(filter(lambda x, m=m: x <= m, [4, 5, 6, 7, 8, 9]))
        if not n:
            n = [m]
        valueStep, requiredRange = (self.valueStep, self.requiredRange)
        if requiredRange and y_max - y_min < requiredRange:
            y1, y2 = find_good_grid(y_min, y_max, n=n, grid=valueStep)[:2]
            if y2 - y1 < requiredRange:
                ym = (y1 + y2) * 0.5
                y1 = min(ym - requiredRange * 0.5, y_min)
                y2 = max(ym + requiredRange * 0.5, y_max)
                if y_min >= 100 and y1 < 100:
                    y2 = y2 + 100 - y1
                    y1 = 100
                elif y_min >= 0 and y1 < 0:
                    y2 = y2 - y1
                    y1 = 0
            self._valueMin, self._valueMax = (y1, y2)
        T, L = ticks(self._valueMin, self._valueMax, split=1, n=n, percent=self.leftAxisPercent, grid=valueStep, labelVOffset=self.labelVOffset)
        abf = self.avoidBoundFrac
        if abf:
            i1 = T[1] - T[0]
            if not isSeq(abf):
                i0 = i1 = i1 * abf
            else:
                i0 = i1 * abf[0]
                i1 = i1 * abf[1]
            _n = getattr(self, '_cValueMin', T[0])
            _x = getattr(self, '_cValueMax', T[-1])
            if _n - T[0] < i0:
                self._valueMin = self._valueMin - i0
            if T[-1] - _x < i1:
                self._valueMax = self._valueMax + i1
            T, L = ticks(self._valueMin, self._valueMax, split=1, n=n, percent=self.leftAxisPercent, grid=valueStep, labelVOffset=self.labelVOffset)
        self._valueMin = T[0]
        self._valueMax = T[-1]
        self._tickValues = T
        if self.labelTextFormat is None:
            self._labelTextFormat = L
        else:
            self._labelTextFormat = self.labelTextFormat
        if abs(self._valueMin - 100) < 1e-06:
            self._calcValueStep()
            vMax, vMin = (self._valueMax, self._valueMin)
            m = max(self.leftAxisOrigShiftIPC * self._valueStep, (vMax - vMin) * self.leftAxisOrigShiftMin / self._length)
            self._valueMin = self._valueMin - m
        if self.leftAxisSkipLL0:
            if isSeq(self.leftAxisSkipLL0):
                for x in self.leftAxisSkipLL0:
                    try:
                        L[x] = ''
                    except IndexError:
                        pass
            L[0] = ''