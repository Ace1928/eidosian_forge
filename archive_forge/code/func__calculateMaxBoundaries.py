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
def _calculateMaxBoundaries(self, colorNamePairs):
    """Calculate the maximum width of some given strings."""
    fontName = self.fontName
    fontSize = self.fontSize
    subCols = self.subCols
    M = [_getWidths(i, m, fontName, fontSize, subCols) for i, m in enumerate(self._getTexts(colorNamePairs))]
    if not M:
        return [0, 0]
    n = max([len(m) for m in M])
    if self.variColumn:
        columnMaximum = self.columnMaximum
        return [_transMax(n, M[r:r + columnMaximum]) for r in range(0, len(M), self.columnMaximum)]
    else:
        return _transMax(n, M)