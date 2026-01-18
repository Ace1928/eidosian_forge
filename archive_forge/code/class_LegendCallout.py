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
class LegendCallout:

    def _legendValues(legend, *args):
        """return a tuple of values from the first function up the stack with isinstance(self,legend)"""
        L = find_locals(lambda L: L.get('self', None) is legend and L or None)
        return tuple([L[a] for a in args])
    _legendValues = staticmethod(_legendValues)

    def _selfOrLegendValues(self, legend, *args):
        L = find_locals(lambda L: L.get('self', None) is legend and L or None)
        return tuple([getattr(self, a, L[a]) for a in args])

    def __call__(self, legend, g, thisx, y, colName):
        col, name = colName