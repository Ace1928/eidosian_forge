import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
class BarChart3D(BarChart):
    _attrMap = AttrMap(BASE=BarChart, theta_x=AttrMapValue(isNumber, desc='dx/dz'), theta_y=AttrMapValue(isNumber, desc='dy/dz'), zDepth=AttrMapValue(isNumber, desc='depth of an individual series'), zSpace=AttrMapValue(isNumber, desc='z gap around series'))
    theta_x = 0.5
    theta_y = 0.5
    zDepth = None
    zSpace = None

    def calcBarPositions(self):
        BarChart.calcBarPositions(self)
        seriesCount = self._seriesCount
        zDepth = self.zDepth
        if zDepth is None:
            zDepth = self.barWidth
        zSpace = self.zSpace
        if zSpace is None:
            zSpace = self.barSpacing
        if self.categoryAxis.style == 'parallel_3d':
            _3d_depth = seriesCount * zDepth + (seriesCount + 1) * zSpace
        else:
            _3d_depth = zDepth + 2 * zSpace
        _3d_depth *= self._normFactor
        self._3d_dx = self.theta_x * _3d_depth
        self._3d_dy = self.theta_y * _3d_depth

    def _calc_z0(self, rowNo):
        zDepth = self.zDepth
        if zDepth is None:
            zDepth = self.barWidth
        zSpace = self.zSpace
        if zSpace is None:
            zSpace = self.barSpacing
        if self.categoryAxis.style == 'parallel_3d':
            z0 = self._normFactor * (rowNo * (zDepth + zSpace) + zSpace)
        else:
            z0 = self._normFactor * zSpace
        return z0

    def _makeBar(self, g, x, y, width, height, rowNo, style):
        zDepth = self.zDepth
        if zDepth is None:
            zDepth = self.barWidth
        zSpace = self.zSpace
        if zSpace is None:
            zSpace = self.barSpacing
        z0 = self._calc_z0(rowNo)
        z1 = z0 + zDepth * self._normFactor
        if width < 0:
            x += width
            width = -width
        x += z0 * self.theta_x
        y += z0 * self.theta_y
        if self._flipXY:
            y += zSpace
        else:
            x += zSpace
        g.add((0, z0, z1, x, y, width, height, rowNo, style))

    def _addBarLabel(self, g, rowNo, colNo, x, y, width, height):
        z0 = self._calc_z0(rowNo)
        zSpace = self.zSpace
        if zSpace is None:
            zSpace = self.barSpacing
        z1 = z0
        x += z0 * self.theta_x
        y += z0 * self.theta_y
        if self._flipXY:
            y += zSpace
        else:
            x += zSpace
        g.add((1, z0, z1, x, y, width, height, rowNo, colNo))

    def makeBars(self):
        from reportlab.graphics.charts.utils3d import _draw_3d_bar
        fg = _FakeGroup(cmp=self._cmpZ)
        self._makeBars(fg, fg)
        fg.sort()
        g = Group()
        theta_x = self.theta_x
        theta_y = self.theta_y
        fg_value = fg.value()
        cAStyle = self.categoryAxis.style
        if cAStyle == 'stacked':
            fg_value.reverse()
        elif cAStyle == 'mixed':
            fg_value = [_[1] for _ in sorted((((t[1], t[2], t[3], t[4]), t) for t in fg_value))]
        for t in fg_value:
            if t[0] == 0:
                z0, z1, x, y, width, height, rowNo, style = t[1:]
                dz = z1 - z0
                _draw_3d_bar(g, x, x + width, y, y + height, dz * theta_x, dz * theta_y, fillColor=style.fillColor, fillColorShaded=None, strokeColor=style.strokeColor, strokeWidth=style.strokeWidth, shading=0.45)
        for t in fg_value:
            if t == 1:
                z0, z1, x, y, width, height, rowNo, colNo = t[1:]
                BarChart._addBarLabel(self, g, rowNo, colNo, x, y, width, height)
        return g