from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Line, Rect, Polygon, PolyLine, \
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.piecharts import WedgeLabel
from reportlab.graphics.widgets.markers import makeMarker, uSymbol2Symbol, isSymbol
class StrandProperty(PropHolder):
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='width'), fillColor=AttrMapValue(isColorOrNone, desc='filling color'), strokeColor=AttrMapValue(isColorOrNone, desc='stroke color'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='dashing pattern, e.g. (3,2)'), symbol=AttrMapValue(EitherOr((isStringOrNone, isSymbol)), desc='Widget placed at data points.', advancedUsage=1), symbolSize=AttrMapValue(isNumber, desc='Symbol size.', advancedUsage=1), name=AttrMapValue(isStringOrNone, desc='Name of the strand.'))

    def __init__(self):
        self.strokeWidth = 1
        self.fillColor = None
        self.strokeColor = STATE_DEFAULTS['strokeColor']
        self.strokeDashArray = STATE_DEFAULTS['strokeDashArray']
        self.symbol = None
        self.symbolSize = 5
        self.name = None