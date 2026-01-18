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
def _getStepsAndLabels(self, xVals):
    if self.dailyFreq:
        xEOM = []
        pm = 0
        px = xVals[0]
        for x in xVals:
            m = x.month()
            if pm != m:
                if pm:
                    xEOM.append(px)
                pm = m
            px = x
        px = xVals[-1]
        if xEOM[-1] != x:
            xEOM.append(px)
        steps, labels = self._xAxisTicker(xEOM)
    else:
        steps, labels = self._xAxisTicker(xVals)
    return (steps, labels)