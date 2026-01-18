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
def _makeSubGrid(self, g, dim=None, parent=None, exclude=[]):
    """this is only called by a container object"""
    if not (getattr(self, 'visibleSubGrid', 0) and self.subTickNum > 0):
        return
    c = self.subGridStrokeColor
    w = self.subGridStrokeWidth or 0
    if not (w and c):
        return
    s = self.subGridStart
    e = self.subGridEnd
    if s is None or e is None:
        if dim and hasattr(dim, '__call__'):
            dim = dim()
        if dim:
            if s is None:
                s = dim[0]
            if e is None:
                e = dim[1]
        else:
            if s is None:
                s = 0
            if e is None:
                e = 0
    if s or e:
        if self.isYAxis:
            offs = self._x
        else:
            offs = self._y
        otv = self._calcSubTicks()
        try:
            self._makeLines(g, s - offs, e - offs, c, w, self.subGridStrokeDashArray, self.subGridStrokeLineJoin, self.subGridStrokeLineCap, self.subGridStrokeMiterLimit, parent=parent, exclude=exclude)
        finally:
            self._tickValues = otv