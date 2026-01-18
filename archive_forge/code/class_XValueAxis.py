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
class XValueAxis(_XTicks, ValueAxis):
    """X/value axis"""
    _attrMap = AttrMap(BASE=ValueAxis, tickUp=AttrMapValue(isNumber, desc='Tick length up the axis.'), tickDown=AttrMapValue(isNumber, desc='Tick length down the axis.'), joinAxis=AttrMapValue(None, desc='Join both axes if true.'), joinAxisMode=AttrMapValue(OneOf('bottom', 'top', 'value', 'points', None), desc="Mode used for connecting axis ('bottom', 'top', 'value', 'points', None)."), joinAxisPos=AttrMapValue(isNumberOrNone, desc='Position at which to join with other axis.'))
    _dataIndex = 0

    def __init__(self, **kw):
        ValueAxis.__init__(self, **kw)
        self.labels.boxAnchor = 'n'
        self.labels.dx = 0
        self.labels.dy = -5
        self.tickUp = 0
        self.tickDown = 5
        self.joinAxis = None
        self.joinAxisMode = None
        self.joinAxisPos = None

    def demo(self):
        self.setPosition(20, 50, 150)
        self.configure([(10, 20, 30, 40, 50)])
        d = Drawing(200, 100)
        d.add(self)
        return d

    def joinToAxis(self, yAxis, mode='bottom', pos=None):
        """Join with y-axis using some mode."""
        _assertYAxis(yAxis)
        if mode == 'bottom':
            self._y = yAxis._y * 1.0
        elif mode == 'top':
            self._y = (yAxis._y + yAxis._length) * 1.0
        elif mode == 'value':
            self._y = yAxis.scale(pos) * 1.0
        elif mode == 'points':
            self._y = pos * 1.0

    def _joinToAxis(self):
        ja = self.joinAxis
        if ja:
            jam = self.joinAxisMode or 'bottom'
            if jam in ('bottom', 'top'):
                self.joinToAxis(ja, mode=jam)
            elif jam in ('value', 'points'):
                self.joinToAxis(ja, mode=jam, pos=self.joinAxisPos)

    def makeAxis(self):
        g = Group()
        self._joinToAxis()
        if not self.visibleAxis:
            return g
        axis = Line(self._x - self.loLLen, self._y, self._x + self._length + self.hiLLen, self._y)
        axis.strokeColor = self.strokeColor
        axis.strokeWidth = self.strokeWidth
        axis.strokeDashArray = self.strokeDashArray
        g.add(axis)
        return g