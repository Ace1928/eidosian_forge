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
def _setupLabel(labelClass, text, radius, cx, cy, angle, car, sar, sty):
    L = labelClass()
    L._text = text
    L.x = cx + radius * car
    L.y = cy + radius * sar
    L._pmv = angle * 180 / pi
    L.boxAnchor = sty.boxAnchor
    L.dx = sty.dx
    L.dy = sty.dy
    L.angle = sty.angle
    L.boxAnchor = sty.boxAnchor
    L.boxStrokeColor = sty.boxStrokeColor
    L.boxStrokeWidth = sty.boxStrokeWidth
    L.boxFillColor = sty.boxFillColor
    L.strokeColor = sty.strokeColor
    L.strokeWidth = sty.strokeWidth
    L.leading = sty.leading
    L.width = sty.width
    L.maxWidth = sty.maxWidth
    L.height = sty.height
    L.textAnchor = sty.textAnchor
    L.visible = sty.visible
    L.topPadding = sty.topPadding
    L.leftPadding = sty.leftPadding
    L.rightPadding = sty.rightPadding
    L.bottomPadding = sty.bottomPadding
    L.fontName = sty.fontName
    L.fontSize = sty.fontSize
    L.fillColor = sty.fillColor
    return L