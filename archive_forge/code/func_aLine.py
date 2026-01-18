from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def aLine(x0, y0, x1, y1):
    if candleKind != 'vertical':
        x0, y0 = (y0, x0)
        x1, y1 = (y1, x1)
    G(shapes.Line(x0, y0, x1, y1, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray))