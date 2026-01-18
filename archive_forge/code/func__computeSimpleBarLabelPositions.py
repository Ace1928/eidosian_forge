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
def _computeSimpleBarLabelPositions(self):
    """Information function, can be called by charts which want to mess with labels"""
    cA, vA = (self.categoryAxis, self.valueAxis)
    if vA:
        vA.joinAxis = cA
    if cA:
        cA.joinAxis = vA
    if self._flipXY:
        cA.setPosition(self._drawBegin(self.x, self.width), self.y, self.height)
    else:
        cA.setPosition(self.x, self._drawBegin(self.y, self.height), self.width)
    cA.configure(self._configureData)
    self.calcBarPositions()
    bars = self.bars
    R = [].append
    BP = self._barPositions
    for rowNo, row in enumerate(BP):
        C = [].append
        for colNo, (x, y, width, height) in enumerate(row):
            if None in (width, height):
                na = self.naLabel
                if na and na.text:
                    na = copy.copy(na)
                    v = self.valueAxis._valueMax <= 0 and -1e-08 or 1e-08
                    if width is None:
                        width = v
                    if height is None:
                        height = v
                    C(self._computeLabelPosition(na.text, na, rowNo, colNo, x, y, width, height))
                else:
                    C(None)
            else:
                text = self._getLabelText(rowNo, colNo)
                if text:
                    C(self._computeLabelPosition(text, self.barLabels[rowNo, colNo], rowNo, colNo, x, y, width, height))
                else:
                    C(None)
        R(C.__self__)
    return R.__self__