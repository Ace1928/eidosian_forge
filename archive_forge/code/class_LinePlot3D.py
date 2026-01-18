from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten, isStr
from reportlab.graphics.shapes import Drawing, Group, Rect, PolyLine, Polygon, _SetKeyWordArgs
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.axes import XValueAxis, YValueAxis, AdjYValueAxis, NormalDateXValueAxis
from reportlab.graphics.charts.utils import *
from reportlab.graphics.widgets.markers import uSymbol2Symbol, makeMarker
from reportlab.graphics.widgets.grids import Grid, DoubleGrid, ShadedPolygon
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.charts.areas import PlotArea
from .utils import FillPairedData
from reportlab.graphics.charts.linecharts import AbstractLineChart
class LinePlot3D(LinePlot):
    _attrMap = AttrMap(BASE=LinePlot, theta_x=AttrMapValue(isNumber, desc='dx/dz'), theta_y=AttrMapValue(isNumber, desc='dy/dz'), zDepth=AttrMapValue(isNumber, desc='depth of an individual series'), zSpace=AttrMapValue(isNumber, desc='z gap around series'))
    theta_x = 0.5
    theta_y = 0.5
    zDepth = 10
    zSpace = 3

    def calcPositions(self):
        LinePlot.calcPositions(self)
        nSeries = self._seriesCount
        zSpace = self.zSpace
        zDepth = self.zDepth
        if self.xValueAxis.style == 'parallel_3d':
            _3d_depth = nSeries * zDepth + (nSeries + 1) * zSpace
        else:
            _3d_depth = zDepth + 2 * zSpace
        self._3d_dx = self.theta_x * _3d_depth
        self._3d_dy = self.theta_y * _3d_depth

    def _calc_z0(self, rowNo):
        zSpace = self.zSpace
        if self.xValueAxis.style == 'parallel_3d':
            z0 = rowNo * (self.zDepth + zSpace) + zSpace
        else:
            z0 = zSpace
        return z0

    def _zadjust(self, x, y, z):
        return (x + z * self.theta_x, y + z * self.theta_y)

    def makeLines(self):
        bubblePlot = getattr(self, '_bubblePlot', None)
        assert not bubblePlot, '_bubblePlot not supported for 3d yet'
        labelFmt = self.lineLabelFormat
        positions = self._positions
        P = list(range(len(positions)))
        if self.reversePlotOrder:
            P.reverse()
        inFill = getattr(self, '_inFill', None)
        assert not inFill, 'inFill not supported for 3d yet'
        zDepth = self.zDepth
        _zadjust = self._zadjust
        theta_x = self.theta_x
        theta_y = self.theta_y
        from reportlab.graphics.charts.linecharts import _FakeGroup
        F = _FakeGroup()
        from reportlab.graphics.charts.utils3d import _make_3d_line_info, find_intersections
        if self.xValueAxis.style != 'parallel_3d':
            tileWidth = getattr(self, '_3d_tilewidth', 1)
            if getattr(self, '_find_intersections', None):
                from copy import copy
                fpositions = list(map(copy, positions))
                I = find_intersections(fpositions, small=tileWidth)
                ic = None
                for i, j, x, y in I:
                    if ic != i:
                        ic = i
                        jc = 0
                    else:
                        jc += 1
                    fpositions[i].insert(j + jc, (x, y))
                tileWidth = None
            else:
                fpositions = positions
        else:
            tileWidth = None
            fpositions = positions
        styleCount = len(self.lines)
        for rowNo in P:
            row = positions[rowNo]
            n = len(row)
            rowStyle = self.lines[rowNo % styleCount]
            rowColor = rowStyle.strokeColor
            dash = getattr(rowStyle, 'strokeDashArray', None)
            z0 = self._calc_z0(rowNo)
            z1 = z0 + zDepth
            if hasattr(rowStyle, 'strokeWidth'):
                width = rowStyle.strokeWidth
            elif hasattr(self.lines, 'strokeWidth'):
                width = self.lines.strokeWidth
            else:
                width = None
            if self.joinedLines:
                if n:
                    frow = fpositions[rowNo]
                    x0, y0 = frow[0]
                    for colNo in range(1, len(frow)):
                        x1, y1 = frow[colNo]
                        _make_3d_line_info(F, x0, x1, y0, y1, z0, z1, theta_x, theta_y, rowColor, fillColorShaded=None, tileWidth=tileWidth, strokeColor=None, strokeWidth=None, strokeDashArray=None, shading=0.1)
                        x0, y0 = (x1, y1)
            if hasattr(rowStyle, 'symbol'):
                uSymbol = rowStyle.symbol
            elif hasattr(self.lines, 'symbol'):
                uSymbol = self.lines.symbol
            else:
                uSymbol = None
            if uSymbol:
                for xy in row:
                    x1, y1 = row[colNo]
                    x1, y1 = _zadjust(x1, y1, z0)
                    symbol = uSymbol2Symbol(uSymbol, xy[0], xy[1], rowColor)
                    if symbol:
                        F.add((1, z0, z0, x1, y1, symbol))
            for colNo in range(n):
                x1, y1 = row[colNo]
                x1, y1 = _zadjust(x1, y1, z0)
                L = self._innerDrawLabel(rowNo, colNo, x1, y1)
                if L:
                    F.add((2, z0, z0, x1, y1, L))
        F.sort()
        g = Group()
        for v in F.value():
            g.add(v[-1])
        return g