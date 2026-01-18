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
def _addWedgeLabel(self, text, angle, labelX, labelY, wedgeStyle, labelClass=None):
    if self.simpleLabels:
        theLabel = String(labelX, labelY, text)
        if not self.sideLabels:
            theLabel.textAnchor = 'middle'
        elif abs(angle) < 90 or (angle > 270 and angle < 450) or -450 < angle < -270:
            theLabel.textAnchor = 'start'
        else:
            theLabel.textAnchor = 'end'
        theLabel._pmv = angle
        theLabel._simple_pointer = 0
    else:
        if labelClass is None:
            labelClass = getattr(self, 'labelClass', WedgeLabel)
        theLabel = labelClass()
        theLabel._pmv = angle
        theLabel.x = labelX
        theLabel.y = labelY
        theLabel.dx = wedgeStyle.label_dx
        if not self.sideLabels:
            theLabel.dy = wedgeStyle.label_dy
            theLabel.boxAnchor = wedgeStyle.label_boxAnchor
        else:
            if wedgeStyle.fontSize is None:
                sideLabels_dy = self.fontSize / 2.5
            else:
                sideLabels_dy = wedgeStyle.fontSize / 2.5
            if wedgeStyle.label_dy is None:
                theLabel.dy = sideLabels_dy
            else:
                theLabel.dy = wedgeStyle.label_dy + sideLabels_dy
            if abs(angle) < 90 or (angle > 270 and angle < 450) or -450 < angle < -270:
                theLabel.boxAnchor = 'w'
            else:
                theLabel.boxAnchor = 'e'
        theLabel.angle = wedgeStyle.label_angle
        theLabel.boxStrokeColor = wedgeStyle.label_boxStrokeColor
        theLabel.boxStrokeWidth = wedgeStyle.label_boxStrokeWidth
        theLabel.boxFillColor = wedgeStyle.label_boxFillColor
        theLabel.strokeColor = wedgeStyle.label_strokeColor
        theLabel.strokeWidth = wedgeStyle.label_strokeWidth
        _text = wedgeStyle.label_text
        if _text is None:
            _text = text
        theLabel._text = _text
        theLabel.leading = wedgeStyle.label_leading
        theLabel.width = wedgeStyle.label_width
        theLabel.maxWidth = wedgeStyle.label_maxWidth
        theLabel.height = wedgeStyle.label_height
        theLabel.textAnchor = wedgeStyle.label_textAnchor
        theLabel.visible = wedgeStyle.label_visible
        theLabel.topPadding = wedgeStyle.label_topPadding
        theLabel.leftPadding = wedgeStyle.label_leftPadding
        theLabel.rightPadding = wedgeStyle.label_rightPadding
        theLabel.bottomPadding = wedgeStyle.label_bottomPadding
        theLabel._simple_pointer = wedgeStyle.label_simple_pointer
    theLabel.fontSize = wedgeStyle.fontSize
    theLabel.fontName = wedgeStyle.fontName
    theLabel.fillColor = wedgeStyle.fontColor
    return theLabel