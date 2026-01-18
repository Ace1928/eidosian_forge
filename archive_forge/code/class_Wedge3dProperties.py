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
class Wedge3dProperties(PropHolder):
    """This holds descriptive information about the wedges in a pie chart.

    It is not to be confused with the 'wedge itself'; this just holds
    a recipe for how to format one, and does not allow you to hack the
    angles.  It can format a genuine Wedge object for you with its
    format method.
    """
    _attrMap = AttrMap(fillColor=AttrMapValue(isColorOrNone, desc=''), fillColorShaded=AttrMapValue(isColorOrNone, desc=''), fontColor=AttrMapValue(isColorOrNone, desc=''), fontName=AttrMapValue(isString, desc=''), fontSize=AttrMapValue(isNumber, desc=''), label_angle=AttrMapValue(isNumber, desc=''), label_bottomPadding=AttrMapValue(isNumber, 'padding at bottom of box'), label_boxAnchor=AttrMapValue(isBoxAnchor, desc=''), label_boxFillColor=AttrMapValue(isColorOrNone, desc=''), label_boxStrokeColor=AttrMapValue(isColorOrNone, desc=''), label_boxStrokeWidth=AttrMapValue(isNumber, desc=''), label_dx=AttrMapValue(isNumber, desc=''), label_dy=AttrMapValue(isNumber, desc=''), label_height=AttrMapValue(isNumberOrNone, desc=''), label_leading=AttrMapValue(isNumberOrNone, desc=''), label_leftPadding=AttrMapValue(isNumber, 'padding at left of box'), label_maxWidth=AttrMapValue(isNumberOrNone, desc=''), label_rightPadding=AttrMapValue(isNumber, 'padding at right of box'), label_simple_pointer=AttrMapValue(isBoolean, 'set to True for simple pointers'), label_strokeColor=AttrMapValue(isColorOrNone, desc=''), label_strokeWidth=AttrMapValue(isNumber, desc=''), label_text=AttrMapValue(isStringOrNone, desc=''), label_textAnchor=AttrMapValue(isTextAnchor, desc=''), label_topPadding=AttrMapValue(isNumber, 'padding at top of box'), label_visible=AttrMapValue(isBoolean, desc='True if the label is to be drawn'), label_width=AttrMapValue(isNumberOrNone, desc=''), labelRadius=AttrMapValue(isNumber, desc=''), popout=AttrMapValue(isNumber, desc=''), shading=AttrMapValue(isNumber, desc=''), strokeColor=AttrMapValue(isColorOrNone, desc=''), strokeColorShaded=AttrMapValue(isColorOrNone, desc=''), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc=''), strokeWidth=AttrMapValue(isNumber, desc=''), visible=AttrMapValue(isBoolean, 'set to false to skip displaying'))

    def __init__(self):
        self.strokeWidth = 0
        self.shading = 0.3
        self.visible = 1
        self.strokeColorShaded = self.fillColorShaded = self.fillColor = None
        self.strokeColor = STATE_DEFAULTS['strokeColor']
        self.strokeDashArray = STATE_DEFAULTS['strokeDashArray']
        self.popout = 0
        self.fontName = STATE_DEFAULTS['fontName']
        self.fontSize = STATE_DEFAULTS['fontSize']
        self.fontColor = STATE_DEFAULTS['fillColor']
        self.labelRadius = 1.2
        self.label_dx = self.label_dy = self.label_angle = 0
        self.label_text = None
        self.label_topPadding = self.label_leftPadding = self.label_rightPadding = self.label_bottomPadding = 0
        self.label_boxAnchor = 'autox'
        self.label_boxStrokeColor = None
        self.label_boxStrokeWidth = 0.5
        self.label_boxFillColor = None
        self.label_strokeColor = None
        self.label_strokeWidth = 0.1
        self.label_leading = self.label_width = self.label_maxWidth = self.label_height = None
        self.label_textAnchor = 'start'
        self.label_visible = 1
        self.label_simple_pointer = 0