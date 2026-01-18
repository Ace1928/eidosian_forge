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