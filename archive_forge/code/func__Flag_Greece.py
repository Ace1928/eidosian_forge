from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Greece(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.gold, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    for stripecounter in range(9, 0, -1):
        stripeheight = s / 9.0
        if not stripecounter % 2 == 0:
            stripecolor = colors.deepskyblue
        else:
            stripecolor = colors.mintcream
        blueorwhiteline = Rect(0, s - stripeheight * stripecounter, width=s * 2, height=stripeheight, fillColor=stripecolor, strokeColor=None, strokeWidth=20)
        g.add(blueorwhiteline)
    bluebox1 = Rect(0, s - stripeheight * 5, width=stripeheight * 5, height=stripeheight * 5, fillColor=colors.deepskyblue, strokeColor=None, strokeWidth=0)
    g.add(bluebox1)
    whiteline1 = Rect(0, s - stripeheight * 3, width=stripeheight * 5, height=stripeheight, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whiteline1)
    whiteline2 = Rect(stripeheight * 2, s - stripeheight * 5, width=stripeheight, height=stripeheight * 5, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whiteline2)
    return g