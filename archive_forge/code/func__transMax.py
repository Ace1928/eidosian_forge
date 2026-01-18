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
def _transMax(n, A):
    X = n * [0]
    m = 0
    for a in A:
        m = max(m, len(a))
        for i, x in enumerate(a):
            X[i] = max(X[i], x)
    X = [0] + X[:m]
    for i in range(m):
        X[i + 1] += X[i]
    return X