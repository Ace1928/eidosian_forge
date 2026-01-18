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
def _fillSide(self, L, i, angle, strokeColor, strokeWidth, fillColor):
    rd = self.rad_dist(angle)
    if rd < self.rad_dist(self._sl3d[i].mid):
        p = [self.CX(i, 0), self.CY(i, 0), self.CX(i, 1), self.CY(i, 1), self.OX(i, angle, 1), self.OY(i, angle, 1), self.OX(i, angle, 0), self.OY(i, angle, 0)]
        L.append((rd, Polygon(p, strokeColor=strokeColor, fillColor=fillColor, strokeWidth=strokeWidth, strokeLineJoin=1)))