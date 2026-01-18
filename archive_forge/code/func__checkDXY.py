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
def _checkDXY(self, ba):
    if ba[0] == 'n':
        if not hasattr(self, '_ody'):
            self._ody = self.dy
            self.dy = -self._ody + self._ydepth_3d