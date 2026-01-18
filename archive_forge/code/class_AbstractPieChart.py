import functools
from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Ellipse, Wedge, String, STATE_DEFAULTS, ArcPath, Polygon, Rect, PolyLine, Line
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.textlabels import Label
from reportlab import cmp
from reportlab.graphics.charts.utils3d import _getShaded, _2rad, _360, _180_pi
class AbstractPieChart(PlotArea):

    def makeSwatchSample(self, rowNo, x, y, width, height):
        baseStyle = self.slices
        styleIdx = rowNo % len(baseStyle)
        style = baseStyle[styleIdx]
        strokeColor = getattr(style, 'strokeColor', getattr(baseStyle, 'strokeColor', None))
        fillColor = getattr(style, 'fillColor', getattr(baseStyle, 'fillColor', None))
        strokeDashArray = getattr(style, 'strokeDashArray', getattr(baseStyle, 'strokeDashArray', None))
        strokeWidth = getattr(style, 'strokeWidth', getattr(baseStyle, 'strokeWidth', None))
        swatchMarker = getattr(style, 'swatchMarker', getattr(baseStyle, 'swatchMarker', None))
        if swatchMarker:
            return uSymbol2Symbol(swatchMarker, x + width / 2.0, y + height / 2.0, fillColor)
        return Rect(x, y, width, height, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray, fillColor=fillColor)

    def getSeriesName(self, i, default=None):
        """return series name i or default"""
        try:
            text = _objStr(self.labels[i])
        except:
            text = default
        if not self.simpleLabels:
            _text = getattr(self.slices[i], 'label_text', '')
            if _text is not None:
                text = _text
        return text