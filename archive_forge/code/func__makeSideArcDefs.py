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
def _makeSideArcDefs(sa, direction):
    sa %= 360
    if 90 <= sa < 270:
        if direction == 'clockwise':
            a = ((0, 90, sa), (1, -90, 90), (0, -360 + sa, -90))
        else:
            a = ((0, sa, 270), (1, 270, 450), (0, 450, 360 + sa))
    else:
        offs = sa >= 270 and 360 or 0
        if direction == 'clockwise':
            a = ((1, offs - 90, sa), (0, offs - 270, offs - 90), (1, -360 + sa, offs - 270))
        else:
            a = ((1, sa, offs + 90), (0, offs + 90, offs + 270), (1, offs + 270, 360 + sa))
    return tuple([a for a in a if a[1] < a[2]])