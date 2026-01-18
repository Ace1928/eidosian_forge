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
class AreaLinePlot(LinePlot):
    """we're given data in the form [(X1,Y11,..Y1M)....(Xn,Yn1,...YnM)]"""

    def __init__(self):
        LinePlot.__init__(self)
        self._inFill = 1
        self.reversePlotOrder = 1
        self.data = [(1, 20, 100, 30), (2, 11, 50, 15), (3, 15, 70, 40)]

    def draw(self):
        try:
            odata = self.data
            n = len(odata)
            m = len(odata[0])
            S = n * [0]
            self.data = []
            for i in range(1, m):
                D = []
                for j in range(n):
                    S[j] = S[j] + odata[j][i]
                    D.append((odata[j][0], S[j]))
                self.data.append(D)
            return LinePlot.draw(self)
        finally:
            self.data = odata