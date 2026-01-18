from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_UK(self):
    s = _size
    g = Group()
    w = s * 2
    g.add(Rect(0, 0, w, s, fillColor=colors.navy, strokeColor=colors.black, strokeWidth=0))
    g.add(Polygon([0, 0, s * 0.225, 0, w, s * (1 - 0.1125), w, s, w - s * 0.225, s, 0, s * 0.1125], fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Polygon([0, s * (1 - 0.1125), 0, s, s * 0.225, s, w, s * 0.1125, w, 0, w - s * 0.225, 0], fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Polygon([0, s - s / 15.0, s - s / 10.0 * 4, s * 0.65, s - s / 10.0 * 3, s * 0.65, 0, s], fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Polygon([0, 0, s - s / 10.0 * 3, s * 0.35, s - s / 10.0 * 2, s * 0.35, s / 10.0, 0], fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Polygon([w, s, s + s / 10.0 * 3, s * 0.65, s + s / 10.0 * 2, s * 0.65, w - s / 10.0, s], fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Polygon([w, s / 15.0, s + s / 10.0 * 4, s * 0.35, s + s / 10.0 * 3, s * 0.35, w, 0], fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Rect(s * 0.42 * 2, 0, width=0.16 * s * 2, height=s, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Rect(0, s * 0.35, width=w, height=s * 0.3, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    g.add(Rect(s * 0.45 * 2, 0, width=0.1 * s * 2, height=s, fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Rect(0, s * 0.4, width=w, height=s * 0.2, fillColor=colors.red, strokeColor=None, strokeWidth=0))
    return g