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
class _AxisG(Widget):

    def _get_line_pos(self, v):
        v = self.scale(v)
        try:
            v = v[0]
        except:
            pass
        return v

    def _cxLine(self, x, start, end):
        x = self._get_line_pos(x)
        return Line(x, self._y + start, x, self._y + end)

    def _cyLine(self, y, start, end):
        y = self._get_line_pos(y)
        return Line(self._x + start, y, self._x + end, y)

    def _cxLine3d(self, x, start, end, _3d_dx, _3d_dy):
        x = self._get_line_pos(x)
        y0 = self._y + start
        y1 = self._y + end
        y0, y1 = (min(y0, y1), max(y0, y1))
        x1 = x + _3d_dx
        return PolyLine([x, y0, x1, y0 + _3d_dy, x1, y1 + _3d_dy], strokeLineJoin=1)

    def _cyLine3d(self, y, start, end, _3d_dx, _3d_dy):
        y = self._get_line_pos(y)
        x0 = self._x + start
        x1 = self._x + end
        x0, x1 = (min(x0, x1), max(x0, x1))
        y1 = y + _3d_dy
        return PolyLine([x0, y, x0 + _3d_dx, y1, x1 + _3d_dx, y1], strokeLineJoin=1)

    def _getLineFunc(self, start, end, parent=None):
        _3d_dx = getattr(parent, '_3d_dx', None)
        if _3d_dx is not None:
            _3d_dy = getattr(parent, '_3d_dy', None)
            f = self.isYAxis and self._cyLine3d or self._cxLine3d
            return lambda v, s=start, e=end, f=f, _3d_dx=_3d_dx, _3d_dy=_3d_dy: f(v, s, e, _3d_dx=_3d_dx, _3d_dy=_3d_dy)
        else:
            f = self.isYAxis and self._cyLine or self._cxLine
            return lambda v, s=start, e=end, f=f: f(v, s, e)

    def _makeLines(self, g, start, end, strokeColor, strokeWidth, strokeDashArray, strokeLineJoin, strokeLineCap, strokeMiterLimit, parent=None, exclude=[], specials={}):
        func = self._getLineFunc(start, end, parent)
        if not hasattr(self, '_tickValues'):
            self._pseudo_configure()
        if exclude:
            exf = self.isYAxis and (lambda l: l.y1 in exclude) or (lambda l: l.x1 in exclude)
        else:
            exf = None
        for t in self._tickValues:
            L = func(t)
            if exf and exf(L):
                continue
            L.strokeColor = strokeColor
            L.strokeWidth = strokeWidth
            L.strokeDashArray = strokeDashArray
            L.strokeLineJoin = strokeLineJoin
            L.strokeLineCap = strokeLineCap
            L.strokeMiterLimit = strokeMiterLimit
            if t in specials:
                for a, v in specials[t].items():
                    setattr(L, a, v)
            g.add(L)

    def makeGrid(self, g, dim=None, parent=None, exclude=[]):
        """this is only called by a container object"""
        c = self.gridStrokeColor
        w = self.gridStrokeWidth or 0
        if w and c and self.visibleGrid:
            s = self.gridStart
            e = self.gridEnd
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
                self._makeLines(g, s - offs, e - offs, c, w, self.gridStrokeDashArray, self.gridStrokeLineJoin, self.gridStrokeLineCap, self.gridStrokeMiterLimit, parent=parent, exclude=exclude, specials=getattr(self, '_gridSpecials', {}))
        self._makeSubGrid(g, dim, parent, exclude=[])

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

    def getGridDims(self, start=None, end=None):
        if start is None:
            start = (self._x, self._y)[self.isYAxis]
        if end is None:
            end = start + self._length
        return (start, end)

    def isYAxis(self):
        if getattr(self, '_dataIndex', None) == 1:
            return True
        acn = self.__class__.__name__
        return acn[0] == 'Y' or acn[:4] == 'AdjY'
    isYAxis = property(isYAxis)

    def isXAxis(self):
        if getattr(self, '_dataIndex', None) == 0:
            return True
        acn = self.__class__.__name__
        return acn[0] == 'X' or acn[:11] == 'NormalDateX'
    isXAxis = property(isXAxis)

    def addAnnotations(self, g, A=None):
        if A is None:
            getattr(self, 'annotations', [])
        for x in A:
            g.add(x(self))

    def _splitAnnotations(self):
        A = getattr(self, 'annotations', [])[:]
        D = {}
        for v in ('early', 'beforeAxis', 'afterAxis', 'beforeTicks', 'afterTicks', 'beforeTickLabels', 'afterTickLabels', 'late'):
            R = [].append
            P = [].append
            for a in A:
                if getattr(a, v, 0):
                    R(a)
                else:
                    P(a)
            D[v] = R.__self__
            A[:] = P.__self__
        D['late'] += A
        return D

    def draw(self):
        g = Group()
        A = self._splitAnnotations()
        self.addAnnotations(g, A['early'])
        if self.visible:
            self.addAnnotations(g, A['beforeAxis'])
            g.add(self.makeAxis())
            self.addAnnotations(g, A['afterAxis'])
            self.addAnnotations(g, A['beforeTicks'])
            g.add(self.makeTicks())
            self.addAnnotations(g, A['afterTicks'])
            self.addAnnotations(g, A['beforeTickLabels'])
            g.add(self.makeTickLabels())
            self.addAnnotations(g, A['afterTickLabels'])
        self.addAnnotations(g, A['late'])
        return g