from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_USA(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.mintcream, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    for stripecounter in range(13, 0, -1):
        stripeheight = s / 13.0
        if not stripecounter % 2 == 0:
            stripecolor = colors.red
        else:
            stripecolor = colors.mintcream
        redorwhiteline = Rect(0, s - stripeheight * stripecounter, width=s * 2, height=stripeheight, fillColor=stripecolor, strokeColor=None, strokeWidth=20)
        g.add(redorwhiteline)
    bluebox = Rect(0, s - stripeheight * 7, width=0.8 * s, height=stripeheight * 7, fillColor=colors.darkblue, strokeColor=None, strokeWidth=0)
    g.add(bluebox)
    lss = s * 0.045
    lss2 = lss / 2.0
    s9 = s / 9.0
    s7 = s / 7.0
    for starxcounter in range(5):
        for starycounter in range(4):
            ls = Star()
            ls.size = lss
            ls.x = 0 - s / 22.0 + lss / 2.0 + s7 + starxcounter * s7
            ls.fillColor = colors.mintcream
            ls.y = s - (starycounter + 1) * s9 + lss2
            g.add(ls)
    for starxcounter in range(6):
        for starycounter in range(5):
            ls = Star()
            ls.size = lss
            ls.x = 0 - s / 22.0 + lss / 2.0 + s / 14.0 + starxcounter * s7
            ls.fillColor = colors.mintcream
            ls.y = s - (starycounter + 1) * s9 + s / 18.0 + lss2
            g.add(ls)
    return g