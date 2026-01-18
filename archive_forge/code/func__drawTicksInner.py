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
def _drawTicksInner(self, tU, tD, g):
    itd = getattr(self, 'innerTickDraw', None)
    if itd:
        itd(self, tU, tD, g)
    elif tU or tD:
        sW = self.actualTickStrokeWidth
        tW = self._tickTweaks
        if tW:
            if tU and (not tD):
                tD = tW * sW
            elif tD and (not tU):
                tU = tW * sW
        self._makeLines(g, tU, -tD, self.actualTickStrokeColor, sW, self.strokeDashArray, self.strokeLineJoin, self.strokeLineCap, self.strokeMiterLimit)