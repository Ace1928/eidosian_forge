import copy
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, OneOf, isString, isColorOrNone,\
from reportlab.lib.attrmap import *
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection, PropHolder
from reportlab.graphics.shapes import Drawing, Group, String, Rect, Line, STATE_DEFAULTS
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.utils import isSeq, find_locals, isStr, asNative
from reportlab.graphics.shapes import _baseGFontName
class LineSwatch(Widget):
    """basically a Line with properties added so it can be used in a LineLegend"""
    _attrMap = AttrMap(x=AttrMapValue(isNumber, desc='x-coordinate for swatch line start point'), y=AttrMapValue(isNumber, desc='y-coordinate for swatch line start point'), width=AttrMapValue(isNumber, desc='length of swatch line'), height=AttrMapValue(isNumber, desc='used for line strokeWidth'), strokeColor=AttrMapValue(isColorOrNone, desc='color of swatch line'), strokeWidth=AttrMapValue(isNumberOrNone, desc='thickness of the swatch'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='dash array for swatch line'))

    def __init__(self):
        from reportlab.lib.colors import red
        self.x = 0
        self.y = 0
        self.width = 20
        self.height = 1
        self.strokeColor = red
        self.strokeDashArray = None
        self.strokeWidth = 1

    def draw(self):
        l = Line(self.x, self.y, self.x + self.width, self.y)
        l.strokeColor = self.strokeColor
        l.strokeDashArray = self.strokeDashArray
        l.strokeWidth = self.strokeWidth
        return l