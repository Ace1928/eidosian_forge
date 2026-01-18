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
def _fakeItemKey(a):
    """t, z0, z1, x, y = a[:5]"""
    return (-a[1], a[3], a[0], -a[4])