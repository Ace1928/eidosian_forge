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
def _findLargestArc(xArcs, side):
    a = [a[1] for a in xArcs if a[0] == side and a[1] is not None]
    if not a:
        return None
    if len(a) > 1:
        a.sort(key=_keyFLA)
    return a[0]