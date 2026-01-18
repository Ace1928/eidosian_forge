from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, \
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, Polygon, PolyLine
from reportlab.graphics.widgets.signsandsymbols import NoEntry
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol, makeMarker
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from .utils import FillPairedData
class SampleHorizontalLineChart(HorizontalLineChart):
    """Sample class overwriting one method to draw additional horizontal lines."""

    def demo(self):
        """Shows basic use of a line chart."""
        drawing = Drawing(200, 100)
        data = [(13, 5, 20, 22, 37, 45, 19, 4), (14, 10, 21, 28, 38, 46, 25, 5)]
        lc = SampleHorizontalLineChart()
        lc.x = 20
        lc.y = 10
        lc.height = 85
        lc.width = 170
        lc.data = data
        lc.strokeColor = colors.white
        lc.fillColor = colors.HexColor(13421772)
        drawing.add(lc)
        return drawing

    def makeBackground(self):
        g = Group()
        g.add(HorizontalLineChart.makeBackground(self))
        valAxis = self.valueAxis
        valTickPositions = valAxis._tickValues
        for y in valTickPositions:
            y = valAxis.scale(y)
            g.add(Line(self.x, y, self.x + self.width, y, strokeColor=self.strokeColor))
        return g