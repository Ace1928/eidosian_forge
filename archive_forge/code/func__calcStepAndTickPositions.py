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
def _calcStepAndTickPositions(self):
    valueStep = getattr(self, '_computedValueStep', None)
    if valueStep:
        del self._computedValueStep
        self._valueStep = valueStep
    else:
        self._calcValueStep()
        valueStep = self._valueStep
    valueMin = self._valueMin
    valueMax = self._valueMax
    fuzz = 1e-08 * valueStep
    rangeRound = self.rangeRound
    i0 = int(float(valueMin) / valueStep)
    v = i0 * valueStep
    if rangeRound in ('both', 'floor'):
        if v > valueMin + fuzz:
            i0 -= 1
    elif v < valueMin - fuzz:
        i0 += 1
    i1 = int(float(valueMax) / valueStep)
    v = i1 * valueStep
    if rangeRound in ('both', 'ceiling'):
        if v < valueMax - fuzz:
            i1 += 1
    elif v > valueMax + fuzz:
        i1 -= 1
    return (valueStep, [i * valueStep for i in range(i0, i1 + 1)])