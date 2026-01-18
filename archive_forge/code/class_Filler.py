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
class Filler:
    """mixin providing simple polygon fill"""
    _attrMap = AttrMap(fillColor=AttrMapValue(isColorOrNone, desc='filler interior color'), strokeColor=AttrMapValue(isColorOrNone, desc='filler edge color'), strokeWidth=AttrMapValue(isNumberOrNone, desc='filler edge width'))

    def __init__(self, **kw):
        self.__dict__ = kw

    def fill(self, lp, g, rowNo, rowColor, points):
        g.add(Polygon(points, fillColor=getattr(self, 'fillColor', rowColor), strokeColor=getattr(self, 'strokeColor', rowColor), strokeWidth=getattr(self, 'strokeWidth', 0.1)))