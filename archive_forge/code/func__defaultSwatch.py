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
def _defaultSwatch(self, x, thisy, dx, dy, fillColor, strokeWidth, strokeColor):
    l = LineSwatch()
    l.x = x
    l.y = thisy
    l.width = dx
    l.height = dy
    l.strokeColor = fillColor
    l.strokeWidth = strokeWidth
    return l