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
def addCrossHair(self, name, xv, yv, strokeColor=colors.black, strokeWidth=1, beforeLines=True):
    from reportlab.graphics.shapes import Group, Line
    annotations = [a for a in getattr(self, 'annotations', []) if getattr(a, 'name', None) != name]

    def annotation(self, xScale, yScale):
        x = xScale(xv)
        y = yScale(yv)
        g = Group()
        xA = xScale.__self__
        g.add(Line(xA._x, y, xA._x + xA._length, y, strokeColor=strokeColor, strokeWidth=strokeWidth))
        yA = yScale.__self__
        g.add(Line(x, yA._y, x, yA._y + yA._length, strokeColor=strokeColor, strokeWidth=strokeWidth))
        return g
    annotation.beforeLines = beforeLines
    annotations.append(annotation)
    self.annotations = annotations