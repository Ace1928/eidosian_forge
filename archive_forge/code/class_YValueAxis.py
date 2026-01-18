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
class YValueAxis(_YTicks, ValueAxis):
    """Y/value axis"""
    _attrMap = AttrMap(BASE=ValueAxis, tickLeft=AttrMapValue(isNumber, desc='Tick length left of the axis.'), tickRight=AttrMapValue(isNumber, desc='Tick length right of the axis.'), joinAxis=AttrMapValue(None, desc='Join both axes if true.'), joinAxisMode=AttrMapValue(OneOf(('left', 'right', 'value', 'points', None)), desc="Mode used for connecting axis ('left', 'right', 'value', 'points', None)."), joinAxisPos=AttrMapValue(isNumberOrNone, desc='Position at which to join with other axis.'))
    _dataIndex = 1

    def __init__(self):
        ValueAxis.__init__(self)
        self.labels.boxAnchor = 'e'
        self.labels.dx = -5
        self.labels.dy = 0
        self.tickRight = 0
        self.tickLeft = 5
        self.joinAxis = None
        self.joinAxisMode = None
        self.joinAxisPos = None

    def demo(self):
        data = [(10, 20, 30, 42)]
        self.setPosition(100, 10, 80)
        self.configure(data)
        drawing = Drawing(200, 100)
        drawing.add(self)
        return drawing

    def joinToAxis(self, xAxis, mode='left', pos=None):
        """Join with x-axis using some mode."""
        _assertXAxis(xAxis)
        if mode == 'left':
            self._x = xAxis._x * 1.0
        elif mode == 'right':
            self._x = (xAxis._x + xAxis._length) * 1.0
        elif mode == 'value':
            self._x = xAxis.scale(pos) * 1.0
        elif mode == 'points':
            self._x = pos * 1.0

    def _joinToAxis(self):
        ja = self.joinAxis
        if ja:
            jam = self.joinAxisMode
            if jam in ('left', 'right'):
                self.joinToAxis(ja, mode=jam)
            elif jam in ('value', 'points'):
                self.joinToAxis(ja, mode=jam, pos=self.joinAxisPos)

    def makeAxis(self):
        g = Group()
        self._joinToAxis()
        if not self.visibleAxis:
            return g
        axis = Line(self._x, self._y - self.loLLen, self._x, self._y + self._length + self.hiLLen)
        axis.strokeColor = self.strokeColor
        axis.strokeWidth = self.strokeWidth
        axis.strokeDashArray = self.strokeDashArray
        g.add(axis)
        return g